"""Drift-triggered training and guarded promotion for the P1 demand model.

Run monthly with ``python -m src.continuous_training``. To roll back, supply both
``--rollback-to`` and ``--expected-current-version``.
"""

import argparse
from datetime import UTC, datetime
import hashlib
from html.parser import HTMLParser
from itertools import pairwise
import json
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from config import (
    HIPERPARAMS_POR_DEFECTO,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_PRODUCTION_ALIAS,
    MLFLOW_REGISTERED_MODELS,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
)
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.ab_testing import bootstrap_delta_rmse_por_dia, metricas_regresion
from src.dataset import procesar_e_integrar_datos, sha256_archivo
from src.features import calcular_features_temporales_y_lags, generar_malla_demanda_horaria
from src.p1_schema import FEATURES, TARGET
from src.tracking import SKOPS_TIPOS_CONFIABLES, obtener_git_commit, repo_con_cambios

URL_DATOS_ABIERTOS = "https://www.mibici.net/es/datos-abiertos/"
HOST_DATOS_ABIERTOS = "www.mibici.net"
MESES_ENTRENAMIENTO = 6
DRIFT_SHARE_TRIGGER = 0.4
MIN_RMSE_IMPROVEMENT = 0.02
MAX_TEST_RMSE = 2.0
MAX_SEGMENT_RMSE_REGRESSION = 0.25
MAX_TRAINING_SECONDS = 900.0
MIN_SEGMENT_ROWS = 100
BOOTSTRAP_REPLICAS = 5000
ARTIFACT_DIR = Path("data/monitoring/continuous-training")
_MONTHLY_CSV = re.compile(r"^/site/assets/files/\d+/datos_abiertos_(\d{4})_(\d{2})\.csv$")
_CATALOG_CSV = re.compile(r"^/site/assets/files/\d+/nomenclatura_(\d{4})_(\d{2})\.csv$")


class _LinksDatos(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


def _url_portal(url: str) -> str:
    absoluta = urljoin(URL_DATOS_ABIERTOS, url)
    parsed = urlparse(absoluta)
    if parsed.scheme != "https" or parsed.hostname != HOST_DATOS_ABIERTOS:
        raise ValueError(f"El portal publicó una URL fuera del dominio HTTPS permitido: {url}")
    return absoluta


def descubrir_archivos_publicos(html: str) -> tuple[list[dict], dict]:
    """Obtiene los CSV mensuales y el catálogo actual desde la página oficial."""
    links = _LinksDatos()
    links.feed(html)
    meses: dict[tuple[int, int], dict] = {}
    catalogos: list[tuple[tuple[int, int], dict]] = []
    for href in links.hrefs:
        parsed = urlparse(urljoin(URL_DATOS_ABIERTOS, href))
        mensual = _MONTHLY_CSV.fullmatch(parsed.path)
        catalogo = _CATALOG_CSV.fullmatch(parsed.path)
        if mensual:
            anio, mes = map(int, mensual.groups())
            if not 1 <= mes <= 12:
                continue
            meses[(anio, mes)] = {
                "periodo": f"{anio:04d}-{mes:02d}",
                "nombre": Path(parsed.path).name,
                "url": _url_portal(href),
            }
        elif catalogo:
            anio, mes = map(int, catalogo.groups())
            if 1 <= mes <= 12:
                catalogos.append(
                    (
                        (anio, mes),
                        {"nombre": Path(parsed.path).name, "url": _url_portal(href)},
                    )
                )

    periodos = sorted(meses)
    if len(periodos) < MESES_ENTRENAMIENTO:
        raise ValueError(
            f"El portal publicó {len(periodos)} meses; se requieren "
            f"{MESES_ENTRENAMIENTO} para entrenar y validar."
        )
    seleccionados = periodos[-MESES_ENTRENAMIENTO:]
    if any(
        despues[0] * 12 + despues[1] - (antes[0] * 12 + antes[1]) != 1
        for antes, despues in pairwise(seleccionados)
    ):
        raise ValueError("Los últimos meses publicados no son consecutivos.")
    if not catalogos:
        raise ValueError("No se encontró un catálogo de estaciones en el portal.")
    return [meses[periodo] for periodo in seleccionados], max(catalogos, key=lambda item: item[0])[
        1
    ]


def _descargar(url: str, destino: Path) -> None:
    """Descarga un recurso HTTPS del dominio permitido sin conservarlo fuera del temporal."""
    if urlparse(url).scheme != "https" or urlparse(url).hostname != HOST_DATOS_ABIERTOS:
        raise ValueError(f"URL de descarga no permitida: {url}")
    solicitud = Request(url, headers={"Accept": "*/*", "User-Agent": "mibici-equipo33"})
    with urlopen(solicitud, timeout=120) as respuesta:  # nosec B310
        destino_final = urlparse(respuesta.geturl())
        if destino_final.scheme != "https" or destino_final.hostname != HOST_DATOS_ABIERTOS:
            raise ValueError(
                f"La descarga redirigió fuera del dominio permitido: {respuesta.geturl()}"
            )
        with destino.open("wb") as archivo:
            while bloque := respuesta.read(1 << 20):
                archivo.write(bloque)


def _cargar_features(archivos: list[dict], catalogo: dict, directorio: Path):
    rutas = []
    manifiesto = []
    for archivo in [*archivos, catalogo]:
        destino = directorio / archivo["nombre"]
        _descargar(archivo["url"], destino)
        manifiesto.append(
            {
                "archivo": archivo["nombre"],
                "url": archivo["url"],
                "bytes": destino.stat().st_size,
                "sha256": sha256_archivo(destino),
            }
        )
        if archivo in archivos:
            rutas.append(destino)

    periodo_inicial = pd.Period(archivos[0]["periodo"], freq="M")
    periodo_actual = pd.Period(archivos[-1]["periodo"], freq="M")
    inicio = periodo_inicial.start_time.strftime("%Y-%m-%d %H:%M:%S")
    fin = periodo_actual.end_time.strftime("%Y-%m-%d %H:%M:%S")
    ruta_viajes = directorio / "viajes_amg_limpios.parquet"
    viajes = procesar_e_integrar_datos(
        rutas,
        directorio / catalogo["nombre"],
        fecha_inicio=inicio,
        fecha_fin=fin,
        ruta_salida=ruta_viajes,
    )
    fecha_ultima = viajes["Inicio_del_viaje"].max()
    if fecha_ultima < periodo_actual.end_time - pd.Timedelta(days=7):
        raise ValueError(
            f"El CSV {archivos[-1]['nombre']} parece incompleto: "
            f"último viaje {fecha_ultima}, periodo {periodo_actual}."
        )

    ids = sorted(set(viajes["Origen_Id"]) | set(viajes["Destino_Id"]))
    malla = generar_malla_demanda_horaria(viajes, ids)
    features = calcular_features_temporales_y_lags(malla)
    if features.empty:
        raise ValueError("La ingeniería de features no produjo filas.")
    if (
        features[[*FEATURES, TARGET]].isna().any().any()
        or not np.isfinite(features[[*FEATURES, TARGET]].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Las features actuales contienen valores nulos o no finitos.")
    data_digest = hashlib.sha256(
        "\n".join(f"{item['archivo']}  {item['sha256']}" for item in manifiesto).encode()
    ).hexdigest()
    return features, manifiesto, data_digest, periodo_actual


def _evaluar_drift(referencia: pd.DataFrame, actual: pd.DataFrame, html_path: Path) -> dict:
    reporte = Report(
        [
            DataDriftPreset(columns=[*FEATURES, TARGET]),
            DataSummaryPreset(),
        ]
    ).run(current_data=actual, reference_data=referencia)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    reporte.save_html(str(html_path))
    resultado = json.loads(reporte.json())
    metricas = resultado.get("metrics", [])
    drift = next(
        (
            m.get("value")
            for m in metricas
            if m.get("metric_name", "").startswith("DriftedColumnsCount(")
        ),
        None,
    )
    if not isinstance(drift, dict) or "share" not in drift or "count" not in drift:
        raise ValueError("Evidently no devolvió DriftedColumnsCount con count y share.")
    return {
        "drifted_columns": int(drift["count"]),
        "drift_share": float(drift["share"]),
        "json": resultado,
    }


def _rmse_por_estacion(real, predicho, estaciones: pd.Series) -> dict[str, dict[str, float | int]]:
    evaluacion = pd.DataFrame(
        {"estacion": estaciones.to_numpy(), "real": np.asarray(real), "predicho": predicho}
    )
    salida = {}
    for estacion, grupo in evaluacion.groupby("estacion"):
        if len(grupo) < MIN_SEGMENT_ROWS:
            continue
        salida[str(int(estacion))] = {
            "filas": len(grupo),
            "rmse": float(np.sqrt(np.mean((grupo["predicho"] - grupo["real"]) ** 2))),
        }
    return salida


def _firma_p1(modelo) -> None:
    firma = modelo.metadata.get_input_schema()
    if firma is None or firma.input_names() != FEATURES:
        raise ValueError("La firma del modelo champion no coincide con las features P1.")


def debe_reentrenar(drift_share: float) -> bool:
    if not 0 <= drift_share <= 1:
        raise ValueError("La proporción de drift debe estar entre 0 y 1.")
    return drift_share >= DRIFT_SHARE_TRIGGER


def evaluar_gates_promocion(
    rmse_candidata: float,
    rmse_champion: float,
    limite_superior_ic_delta: float,
    max_delta_rmse_estacion: float,
    segundos_entrenamiento: float,
) -> dict[str, bool]:
    """Evalúa gates de calidad estadística, errores segmentados y costo de entrenamiento."""
    valores = (
        rmse_candidata,
        rmse_champion,
        limite_superior_ic_delta,
        max_delta_rmse_estacion,
        segundos_entrenamiento,
    )
    if not all(np.isfinite(valor) for valor in valores):
        raise ValueError("Las métricas para promover deben ser finitas.")
    return {
        "absolute_quality": rmse_candidata <= MAX_TEST_RMSE,
        "relative_quality": rmse_candidata <= rmse_champion * (1 - MIN_RMSE_IMPROVEMENT),
        "statistical_quality": limite_superior_ic_delta < 0,
        "segment_quality": max_delta_rmse_estacion <= MAX_SEGMENT_RMSE_REGRESSION,
        "training_cost": 0 <= segundos_entrenamiento <= MAX_TRAINING_SECONDS,
    }


def _promocionar(client: MlflowClient, nombre: str, version: str, version_actual: str) -> None:
    alias_actual = client.get_model_version_by_alias(nombre, MLFLOW_PRODUCTION_ALIAS)
    if str(alias_actual.version) != str(version_actual):
        raise RuntimeError(
            f"El alias cambió durante la evaluación: esperaba {version_actual}, "
            f"ahora apunta a {alias_actual.version}; no se promoverá."
        )
    client.set_registered_model_alias(nombre, MLFLOW_PRODUCTION_ALIAS, version)
    alias_nuevo = client.get_model_version_by_alias(nombre, MLFLOW_PRODUCTION_ALIAS)
    if str(alias_nuevo.version) != str(version):
        client.set_registered_model_alias(nombre, MLFLOW_PRODUCTION_ALIAS, version_actual)
        restaurado = client.get_model_version_by_alias(nombre, MLFLOW_PRODUCTION_ALIAS)
        if str(restaurado.version) != str(version_actual):
            raise RuntimeError(
                f"No se verificó la promoción ni la restauración del alias: "
                f"esperado {version_actual}, actual {restaurado.version}."
            )
        raise RuntimeError(
            f"No se verificó la promoción de {version}; se restauró el alias "
            f"{MLFLOW_PRODUCTION_ALIAS} a {version_actual}."
        )


def _guardar_resultado(resultado: dict, directorio: Path) -> Path:
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / "latest.json"
    ruta.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    data_version = resultado.get("data_version", "rollback")
    (directorio / f"{data_version}.json").write_text(
        ruta.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return ruta


def _registrar_candidata(
    client: MlflowClient, run_id: str, data_version: str, data_digest: str, decision: dict
) -> str:
    nombre = MLFLOW_REGISTERED_MODELS["p1"]
    encontrada = next(
        (
            version
            for version in client.search_model_versions(f"name='{nombre}'")
            if version.run_id == run_id
        ),
        None,
    )
    registrada = encontrada or mlflow.register_model(
        f"runs:/{run_id}/modelo", nombre, await_registration_for=300
    )
    version_id = str(registrada.version)
    version = client.get_model_version(nombre, version_id)
    if version.status != "READY" or version.run_id != run_id:
        raise ValueError(f"La versión candidata {nombre}/{version_id} no está READY.")
    run = client.get_run(run_id)
    if run.data.tags.get("git_dirty") != "False":
        raise ValueError("No se registra una candidata entrenada con un árbol Git sucio.")
    tags_version = {
        "question": "P1",
        "stage_of_work": "candidate",
        "model_family": "HistGradientBoostingRegressor",
        "source_experiment": MLFLOW_EXPERIMENT_NAME,
        "source_experiment_id": run.info.experiment_id,
        "source_run_id": run_id,
        "git_commit": run.data.tags["git_commit"],
        "git_dirty": run.data.tags["git_dirty"],
        "data_version": data_version,
        "data_sha256": data_digest,
        "promotion_decision": decision["decision"],
        "promotion_reason": decision["reason"],
    }
    tags_version["promotion_decision"] = "evaluated"
    for key, value in tags_version.items():
        client.set_model_version_tag(nombre, version_id, key, str(value))
    return version_id


def _fingerprint(datos_sha256: str, commit: str) -> str:
    return hashlib.sha256(f"{datos_sha256}\n{commit}".encode()).hexdigest()


def _run_id_cubierto(client: MlflowClient, experiment_id: str, fingerprint: str) -> bool:
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.ct_fingerprint = '{fingerprint}'",
        max_results=1000,
    )
    return any(run.info.status == "FINISHED" for run in runs)


def _entrenar_candidato(
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_version: str,
    data_digest: str,
    fingerprint: str,
    drift_share: float,
) -> tuple[HistGradientBoostingRegressor, str, dict, float]:
    modelo = HistGradientBoostingRegressor(
        **HIPERPARAMS_POR_DEFECTO["p1"]["histgb"],
        random_state=RANDOM_STATE,
    )
    inicio = time.monotonic()
    with mlflow.start_run(run_name=f"P1-CT-{data_version}", nested=True) as run:
        mlflow.set_tags(
            {
                "question": "P1",
                "stage_of_work": "candidate",
                "model_family": "HistGradientBoostingRegressor",
                "git_commit": obtener_git_commit(),
                "git_dirty": str(repo_con_cambios()),
                "data_version": data_version,
                "ct_data_sha256": data_digest,
                "ct_fingerprint": fingerprint,
                "monitoring_trigger": "evidently_drift",
            }
        )
        mlflow.log_params(
            {
                "data_version": data_version,
                "data_sha256": data_digest,
                "drift_share": drift_share,
                "training_months": MESES_ENTRENAMIENTO - 1,
                "test_month": data_version.split("-")[2],
                "n_train": len(train),
                "n_test": len(test),
                "n_stations": int(test["Origen_Id"].nunique()),
                "python_random_state": RANDOM_STATE,
            }
        )
        modelo.fit(train[FEATURES], train[TARGET])
        predicho = np.clip(modelo.predict(test[FEATURES]), 0, None)
        metricas = metricas_regresion(test[TARGET], predicho)
        mlflow.log_metrics({f"test_{key}": value for key, value in metricas.items()})
        signature = infer_signature(train[FEATURES], modelo.predict(train[FEATURES]))
        mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            signature=signature,
            input_example=train[FEATURES].iloc[:3],
            skops_trusted_types=SKOPS_TIPOS_CONFIABLES,
        )
        run_id = run.info.run_id
    return modelo, run_id, metricas, time.monotonic() - inicio


def ejecutar_ciclo(directorio_evidencia: Path = ARTIFACT_DIR) -> dict:
    """Descarga datos recientes, detecta drift y promueve únicamente si supera los gates."""
    commit = obtener_git_commit()
    if not commit or commit == "desconocido" or repo_con_cambios():
        raise RuntimeError(
            "Continuous Training requiere un commit identificable y un árbol Git limpio."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"No existe el experimento MLflow {MLFLOW_EXPERIMENT_NAME!r}.")
    nombre_modelo = MLFLOW_REGISTERED_MODELS["p1"]
    vigente = client.get_model_version_by_alias(nombre_modelo, MLFLOW_PRODUCTION_ALIAS)
    vigente_id = str(vigente.version)
    champion = mlflow.pyfunc.load_model(f"models:/{nombre_modelo}@{MLFLOW_PRODUCTION_ALIAS}")
    _firma_p1(champion)

    solicitud = Request(URL_DATOS_ABIERTOS, headers={"User-Agent": "mibici-equipo33"})
    with urlopen(solicitud, timeout=60) as respuesta:  # nosec B310
        destino_final = urlparse(respuesta.geturl())
        if destino_final.scheme != "https" or destino_final.hostname != HOST_DATOS_ABIERTOS:
            raise ValueError(
                f"El portal redirigió fuera del dominio permitido: {respuesta.geturl()}"
            )
        pagina = respuesta.read().decode("utf-8")
    archivos, catalogo = descubrir_archivos_publicos(pagina)
    with tempfile.TemporaryDirectory(prefix="mibici-ct-") as temporal:
        features, manifiesto, data_digest, periodo = _cargar_features(
            archivos, catalogo, Path(temporal)
        )
        data_version = f"mibici-ct-{periodo.strftime('%Y%m')}-{data_digest[:16]}"
        fingerprint = _fingerprint(data_digest, commit)
        if _run_id_cubierto(client, experiment.experiment_id, fingerprint):
            resultado = {
                "status": "already_processed",
                "decision": "mantener",
                "reason": "Estos hashes de datos ya fueron evaluados por Continuous Training.",
                "data_version": data_version,
                "data_sha256": data_digest,
                "champion_version": vigente_id,
            }
            _guardar_resultado(resultado, Path(directorio_evidencia))
            return resultado

        corte = periodo.start_time
        train = features.loc[features["timestamp_hora"] < corte].copy()
        test = features.loc[
            features["timestamp_hora"].ge(corte) & features["timestamp_hora"].le(periodo.end_time),
        ].copy()
        if train.empty or test.empty or test["timestamp_hora"].dt.normalize().nunique() < 2:
            raise ValueError("La ventana temporal no permite entrenar y validar por día.")

        actualidades = Path(temporal) / "drift.html"
        drift = _evaluar_drift(train, test, actualidades)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"P1-CT-monitor-{data_version}") as monitor_run:
            mlflow.set_tags(
                {
                    "question": "P1",
                    "stage_of_work": "monitoring",
                    "ct_data_sha256": data_digest,
                    "ct_fingerprint": fingerprint,
                    "data_version": data_version,
                    "monitoring_trigger": "evidently_drift",
                }
            )
            mlflow.log_params(
                {
                    "drift_share_trigger": DRIFT_SHARE_TRIGGER,
                    "rmse_min_improvement": MIN_RMSE_IMPROVEMENT,
                    "max_test_rmse": MAX_TEST_RMSE,
                    "max_training_seconds": MAX_TRAINING_SECONDS,
                    "max_segment_rmse_regression": MAX_SEGMENT_RMSE_REGRESSION,
                    "current_champion_version": vigente_id,
                }
            )
            mlflow.log_metrics(
                {
                    "drift_share": drift["drift_share"],
                    "drifted_columns": drift["drifted_columns"],
                }
            )
            mlflow.log_artifact(str(actualidades), artifact_path="drift")
            (Path(temporal) / "drift.json").write_text(
                json.dumps(drift["json"], indent=2, ensure_ascii=False), encoding="utf-8"
            )
            mlflow.log_artifact(str(Path(temporal) / "drift.json"), artifact_path="drift")
            manifiesto_path = Path(temporal) / "data-manifest.json"
            manifiesto_path.write_text(
                json.dumps(
                    {
                        "data_version": data_version,
                        "data_sha256": data_digest,
                        "files": manifiesto,
                        "reference_period": f"{train['timestamp_hora'].min().isoformat()} to "
                        f"{train['timestamp_hora'].max().isoformat()}",
                        "current_period": f"{test['timestamp_hora'].min().isoformat()} to "
                        f"{test['timestamp_hora'].max().isoformat()}",
                        "rows_train": len(train),
                        "rows_test": len(test),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            manifiesto_sha256 = sha256_archivo(manifiesto_path)
            mlflow.log_artifact(str(manifiesto_path), artifact_path="provenance")

            if not debe_reentrenar(drift["drift_share"]):
                resultado = {
                    "status": "completed",
                    "decision": "mantener",
                    "reason": "La proporción de variables con drift no alcanzó el umbral.",
                    "data_version": data_version,
                    "data_sha256": data_digest,
                    "source_files": manifiesto,
                    "manifest_sha256": manifiesto_sha256,
                    "drift_report_sha256": {
                        "html": sha256_archivo(actualidades),
                        "json": sha256_archivo(Path(temporal) / "drift.json"),
                    },
                    "drift": {
                        "drifted_columns": drift["drifted_columns"],
                        "drift_share": drift["drift_share"],
                    },
                    "champion_version": vigente_id,
                    "monitor_run_id": monitor_run.info.run_id,
                }
                mlflow.set_tag("ct_decision", resultado["decision"])
            else:
                candidate, run_id, metricas, segundos = _entrenar_candidato(
                    train,
                    test,
                    data_version,
                    data_digest,
                    fingerprint,
                    drift["drift_share"],
                )
                real = test[TARGET].to_numpy()
                pred_candidato = np.clip(candidate.predict(test[FEATURES]), 0, None)
                pred_vigente = np.clip(champion.predict(test[FEATURES]), 0, None)
                metricas_vigente = metricas_regresion(real, pred_vigente)
                intervalo = bootstrap_delta_rmse_por_dia(
                    real,
                    pred_vigente,
                    pred_candidato,
                    test["timestamp_hora"],
                    n_replicas=BOOTSTRAP_REPLICAS,
                    random_state=RANDOM_STATE,
                )
                errores_vigente = _rmse_por_estacion(real, pred_vigente, test["Origen_Id"])
                errores_candidato = _rmse_por_estacion(real, pred_candidato, test["Origen_Id"])
                deltas_segmentos = {
                    estacion: errores_candidato[estacion]["rmse"]
                    - errores_vigente[estacion]["rmse"]
                    for estacion in errores_vigente.keys() & errores_candidato.keys()
                }
                if not deltas_segmentos:
                    raise ValueError(
                        f"No hay estaciones con al menos {MIN_SEGMENT_ROWS} filas "
                        "comparables para evaluar errores segmentados."
                    )
                max_delta_segmento = max(deltas_segmentos.values(), default=0.0)
                gates = evaluar_gates_promocion(
                    metricas["rmse"],
                    metricas_vigente["rmse"],
                    intervalo["ic_95_superior"],
                    max_delta_segmento,
                    segundos,
                )
                calidad = all(
                    gates[key]
                    for key in (
                        "absolute_quality",
                        "relative_quality",
                        "statistical_quality",
                        "segment_quality",
                    )
                )
                costo = gates["training_cost"]
                decision = {
                    "decision": "promover" if calidad and costo else "mantener",
                    "reason": (
                        "Cumple los gates de calidad estadística, errores por estación y costo de "
                        "entrenamiento."
                        if calidad and costo
                        else "La candidata no cumplió todos los gates; se conserva champion."
                    ),
                    "quality_gate": calidad,
                    "cost_gate": costo,
                    "quality_checks": gates,
                    "candidate_run_id": run_id,
                    "candidate_rmse": metricas["rmse"],
                    "champion_rmse": metricas_vigente["rmse"],
                    "bootstrap_rmse_delta_candidate_minus_champion": intervalo,
                    "max_station_rmse_delta": max_delta_segmento,
                    "station_rmse": {
                        estacion: {
                            "champion": errores_vigente[estacion]["rmse"],
                            "candidate": errores_candidato[estacion]["rmse"],
                            "delta": deltas_segmentos[estacion],
                            "rows": errores_candidato[estacion]["filas"],
                        }
                        for estacion in deltas_segmentos
                    },
                    "training_seconds": segundos,
                    "thresholds": {
                        "minimum_rmse_improvement": MIN_RMSE_IMPROVEMENT,
                        "maximum_test_rmse": MAX_TEST_RMSE,
                        "maximum_station_rmse_regression": MAX_SEGMENT_RMSE_REGRESSION,
                        "maximum_training_seconds": MAX_TRAINING_SECONDS,
                    },
                }
                version = _registrar_candidata(client, run_id, data_version, data_digest, decision)
                decision["registered_candidate_version"] = version
                if decision["decision"] == "promover":
                    _promocionar(client, nombre_modelo, version, vigente_id)
                    decision["champion_version"] = version
                else:
                    decision["champion_version"] = vigente_id
                client.set_model_version_tag(
                    nombre_modelo, version, "promotion_decision", decision["decision"]
                )
                client.set_model_version_tag(
                    nombre_modelo, version, "promotion_reason", decision["reason"]
                )
                mlflow.log_metrics(
                    {
                        "candidate_test_rmse": metricas["rmse"],
                        "champion_test_rmse": metricas_vigente["rmse"],
                        "training_seconds": segundos,
                        "bootstrap_rmse_delta_upper_95": intervalo["ic_95_superior"],
                        "max_station_rmse_delta": max_delta_segmento,
                    }
                )
                resultado = {
                    "status": "completed",
                    "data_version": data_version,
                    "data_sha256": data_digest,
                    "source_files": manifiesto,
                    "manifest_sha256": manifiesto_sha256,
                    "drift_report_sha256": {
                        "html": sha256_archivo(actualidades),
                        "json": sha256_archivo(Path(temporal) / "drift.json"),
                    },
                    "drift": {
                        "drifted_columns": drift["drifted_columns"],
                        "drift_share": drift["drift_share"],
                    },
                    "champion_before": vigente_id,
                    "monitor_run_id": monitor_run.info.run_id,
                    **decision,
                }
                mlflow.set_tag("ct_decision", resultado["decision"])
                mlflow.log_dict(resultado, "decision.json")
            ruta_evidencia = _guardar_resultado(resultado, Path(directorio_evidencia))
            mlflow.log_artifact(str(ruta_evidencia), artifact_path="decision")
        return resultado


def revertir_alias(
    version_objetivo: str,
    version_esperada: str,
    directorio_evidencia: Path = ARTIFACT_DIR,
) -> dict:
    """Revierte champion solo si sigue apuntando a la versión confirmada por el operador."""
    if not version_objetivo or not version_esperada:
        raise ValueError("La versión destino y la versión vigente esperada son obligatorias.")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    nombre = MLFLOW_REGISTERED_MODELS["p1"]
    actual = client.get_model_version_by_alias(nombre, MLFLOW_PRODUCTION_ALIAS)
    if str(actual.version) != version_esperada:
        raise RuntimeError(
            f"Reversión cancelada: champion apunta a {actual.version}, "
            f"no a la versión esperada {version_esperada}."
        )
    destino = client.get_model_version(nombre, version_objetivo)
    if destino.status != "READY" or destino.tags.get("question") != "P1":
        raise ValueError(f"La versión destino {version_objetivo} no es una versión P1 READY.")
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version_objetivo}")
    _firma_p1(modelo)
    vigente_previo = client.get_model_version_by_alias(nombre, MLFLOW_PRODUCTION_ALIAS)
    if str(vigente_previo.version) != version_esperada:
        raise RuntimeError(
            f"Reversión cancelada: champion cambió a {vigente_previo.version} durante la validación."
        )
    client.set_registered_model_alias(nombre, MLFLOW_PRODUCTION_ALIAS, version_objetivo)
    verificacion = client.get_model_version_by_alias(nombre, MLFLOW_PRODUCTION_ALIAS)
    if str(verificacion.version) != version_objetivo:
        client.set_registered_model_alias(nombre, MLFLOW_PRODUCTION_ALIAS, version_esperada)
        restaurado = client.get_model_version_by_alias(nombre, MLFLOW_PRODUCTION_ALIAS)
        if str(restaurado.version) != version_esperada:
            raise RuntimeError(
                f"No se verificó la reversión ni la restauración de champion: "
                f"esperado {version_esperada}, actual {restaurado.version}."
            )
        raise RuntimeError("No se verificó la reversión; se restauró el alias previo.")
    resultado = {
        "status": "rolled_back",
        "decision": "revertir",
        "previous_champion_version": version_esperada,
        "champion_version": version_objetivo,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    _guardar_resultado(resultado, Path(directorio_evidencia))
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback-to", help="Versión READY de P1 a asignar al alias champion.")
    parser.add_argument(
        "--expected-current-version",
        help="Versión a la que debe apuntar champion antes de la reversión.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    if bool(args.rollback_to) != bool(args.expected_current_version):
        parser.error("--rollback-to y --expected-current-version deben usarse juntos.")
    if args.rollback_to:
        resultado = revertir_alias(
            args.rollback_to,
            args.expected_current_version,
            args.artifact_dir,
        )
    else:
        resultado = ejecutar_ciclo(args.artifact_dir)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
