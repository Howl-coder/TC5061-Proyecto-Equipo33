"""Compara versiones registradas de P1 sobre el periodo actual.

Uso:
    python -m src.ab_testing --version-a 1 --version-b 2
"""

import argparse
import json
from pathlib import Path

from config import MLFLOW_REGISTERED_MODELS, MLFLOW_TRACKING_URI, REPORTS_DIR
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.monitoring_data import COLUMNA_TIEMPO, RUTA_SALIDA, verificar_datasets_monitoreo
from src.p1_schema import FEATURES, TARGET

RUTA_RESULTADOS = REPORTS_DIR / "monitoring" / "ab_test_p1.json"


def _validar_vectores(y_real, y_predicho) -> tuple[np.ndarray, np.ndarray]:
    real = np.asarray(y_real, dtype=float).reshape(-1)
    predicho = np.asarray(y_predicho, dtype=float).reshape(-1)
    if real.size == 0 or real.shape != predicho.shape:
        raise ValueError(
            "Los vectores de valores reales y predichos deben tener igual tamaño > 0."
        )
    if not np.isfinite(real).all() or not np.isfinite(predicho).all():
        raise ValueError("Los vectores de valores reales y predichos deben ser finitos.")
    return real, predicho


def metricas_regresion(y_real, y_predicho) -> dict[str, float]:
    """Calcula RMSE, MAE, R² y sesgo medio sobre los vectores recibidos tal cual.

    El recorte de predicciones negativas a 0 es responsabilidad del llamador
    (ver ``comparar_versiones``).
    """
    real, predicho = _validar_vectores(y_real, y_predicho)
    return {
        "rmse": float(np.sqrt(mean_squared_error(real, predicho))),
        "mae": float(mean_absolute_error(real, predicho)),
        "r2": float(r2_score(real, predicho)),
        "sesgo_pred_menos_real": float(np.mean(predicho - real)),
    }


def bootstrap_delta_rmse_por_dia(
    y_real,
    pred_a,
    pred_b,
    timestamps,
    n_replicas: int = 5000,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Estima por bootstrap de días el cambio de RMSE B - A con pares alineados."""
    real, a = _validar_vectores(y_real, pred_a)
    _, b = _validar_vectores(y_real, pred_b)
    fechas = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
    if len(fechas) != len(real) or fechas.isna().any():
        raise ValueError("Debe haber una fecha válida por cada predicción.")
    if n_replicas < 1:
        raise ValueError("n_replicas debe ser al menos 1.")

    dias = fechas.normalize().to_numpy()
    dias_unicos = pd.unique(dias)
    if len(dias_unicos) < 2:
        raise ValueError("El bootstrap requiere al menos dos bloques diarios.")

    errores_a = (a - real) ** 2
    errores_b = (b - real) ** 2
    suma_a = np.array([errores_a[dias == dia].sum() for dia in dias_unicos])
    suma_b = np.array([errores_b[dias == dia].sum() for dia in dias_unicos])
    conteos = np.array([(dias == dia).sum() for dia in dias_unicos])

    rng = np.random.default_rng(random_state)
    muestras = rng.integers(0, len(dias_unicos), size=(n_replicas, len(dias_unicos)))
    n_filas = conteos[muestras].sum(axis=1)
    rmse_a = np.sqrt(suma_a[muestras].sum(axis=1) / n_filas)
    rmse_b = np.sqrt(suma_b[muestras].sum(axis=1) / n_filas)
    delta = rmse_b - rmse_a

    return {
        "delta_observado_rmse_b_menos_a": float(
            np.sqrt(errores_b.mean()) - np.sqrt(errores_a.mean())
        ),
        "ic_95_inferior": float(np.quantile(delta, 0.025)),
        "ic_95_superior": float(np.quantile(delta, 0.975)),
        "fraccion_replicas_b_peor": float(np.mean(delta > 0)),
        "bloques_dia": len(dias_unicos),
        "replicas": int(n_replicas),
        "random_state": int(random_state),
    }


def comparar_versiones(
    version_a: str,
    version_b: str,
    datos_actuales: pd.DataFrame,
    n_replicas: int = 5000,
    random_state: int = 42,
) -> dict:
    """Evalúa dos versiones READY del mismo modelo sobre las mismas filas."""
    if version_a == version_b:
        raise ValueError("Las versiones A y B deben ser distintas.")
    columnas = [COLUMNA_TIEMPO, *FEATURES, TARGET]
    faltantes = [col for col in columnas if col not in datos_actuales.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el conjunto actual: {faltantes}")
    if datos_actuales.empty:
        raise ValueError("El conjunto actual está vacío.")

    nombre = MLFLOW_REGISTERED_MODELS["p1"]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    x = datos_actuales[FEATURES]
    y = datos_actuales[TARGET].to_numpy()
    metricas = {}
    metadatos = {}
    predicciones = {}

    for version_id in (version_a, version_b):
        version = client.get_model_version(nombre, version_id)
        if version.status != "READY":
            raise ValueError(f"El modelo {nombre} versión {version_id} no está READY.")
        if version.tags.get("question") != "P1":
            raise ValueError(
                f"El modelo {nombre} versión {version_id} no está etiquetado como P1."
            )

        modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version_id}")
        firma = modelo.metadata.get_input_schema()
        if firma is None or firma.input_names() != FEATURES:
            raise ValueError(f"La firma de {nombre} versión {version_id} no coincide con P1.")
        pred = np.clip(np.asarray(modelo.predict(x)).reshape(-1), 0, None)
        predicciones[version_id] = pred
        metricas[version_id] = metricas_regresion(y, pred)

        run = client.get_run(version.run_id)
        metadatos[version_id] = {
            "run_id": version.run_id,
            "stage_of_work": version.tags.get("stage_of_work"),
            "git_dirty": version.tags.get("git_dirty"),
            "git_commit": version.tags.get("git_commit"),
            "data_version": version.tags.get("data_version"),
            "hiperparametros": {
                key: value
                for key, value in run.data.params.items()
                if key
                in {
                    "max_iter",
                    "learning_rate",
                    "max_depth",
                    "min_samples_leaf",
                    "l2_regularization",
                }
            },
        }

    intervalo = bootstrap_delta_rmse_por_dia(
        y,
        predicciones[version_a],
        predicciones[version_b],
        datos_actuales[COLUMNA_TIEMPO],
        n_replicas=n_replicas,
        random_state=random_state,
    )
    return {
        "registered_model": nombre,
        "version_a": version_a,
        "version_b": version_b,
        "evaluacion": {
            "filas": len(datos_actuales),
            "estaciones": int(datos_actuales["Origen_Id"].nunique())
            if "Origen_Id" in datos_actuales
            else None,
            "desde": pd.Timestamp(datos_actuales[COLUMNA_TIEMPO].min()).isoformat(),
            "hasta": pd.Timestamp(datos_actuales[COLUMNA_TIEMPO].max()).isoformat(),
            "target": TARGET,
            "features": FEATURES,
        },
        "modelos": metadatos,
        "metricas": metricas,
        "bootstrap_rmse_b_menos_a": intervalo,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-a", default="1", help="Versión vigente del modelo P1.")
    parser.add_argument("--version-b", default="2", help="Versión candidata del modelo P1.")
    parser.add_argument("--n-replicas", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=RUTA_RESULTADOS)
    args = parser.parse_args()
    if args.n_replicas < 1:
        parser.error("--n-replicas debe ser al menos 1.")

    ruta_datos = RUTA_SALIDA / "actual_p1.parquet"
    if not ruta_datos.exists():
        raise FileNotFoundError(
            f"No existe {ruta_datos}. Ejecuta primero `python -m src.monitoring_data`."
        )
    procedencia = verificar_datasets_monitoreo()
    datos = pd.read_parquet(ruta_datos)
    resultados = comparar_versiones(
        args.version_a,
        args.version_b,
        datos,
        n_replicas=args.n_replicas,
        random_state=args.seed,
    )
    resultados["procedencia_datasets"] = procedencia
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for version in (args.version_a, args.version_b):
        print(f"Versión {version}: {resultados['metricas'][version]}")
    print(f"Delta RMSE B-A e IC 95%: {resultados['bootstrap_rmse_b_menos_a']}")
    print(f"Resultados: {args.output}")


if __name__ == "__main__":
    main()
