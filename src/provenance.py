"""Comprobación de procedencia de los datos intermedios y procesados."""

import json
import os
from pathlib import Path

from config import BASE_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

from src.dataset import sha256_archivo

INTERIM = INTERIM_DATA_DIR / "viajes_amg_limpios.parquet"
FEATURES = PROCESSED_DATA_DIR / "features_demanda_p1.parquet"


def _metadatos(ruta: Path, etapa: str, data_sha256: str) -> dict:
    codigos = [BASE_DIR / "config.py", BASE_DIR / "src" / "dataset.py"]
    if etapa == "features":
        codigos.append(BASE_DIR / "src" / "features.py")
    elif etapa != "ingest":
        raise ValueError(f"Etapa desconocida: {etapa}")
    resultado = {
        "etapa": etapa,
        "data_sha256": data_sha256,
        "archivo_sha256": sha256_archivo(ruta),
        "codigo_sha256": {p.name: sha256_archivo(p) for p in codigos},
    }
    if etapa == "features":
        resultado["interim_sha256"] = sha256_archivo(INTERIM)
    return resultado


def _ruta_metadatos(ruta: Path) -> Path:
    return ruta.with_suffix(ruta.suffix + ".metadata.json")


def guardar_procedencia(ruta: Path, etapa: str, data_sha256: str) -> None:
    metadatos = _metadatos(ruta, etapa, data_sha256)
    _ruta_metadatos(ruta).write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def registrar_procedencia_mlflow(ruta: Path, etapa: str, data_sha256: str) -> None:
    if not os.environ.get("MLFLOW_RUN_ID"):
        return
    import mlflow

    metadatos = verificar_procedencia(ruta, etapa, data_sha256)
    with mlflow.start_run():
        mlflow.set_tag("etapa_pipeline", etapa)
        mlflow.log_params(
            {"data_sha256": data_sha256, "archivo_sha256": metadatos["archivo_sha256"]}
        )
        mlflow.log_artifact(str(_ruta_metadatos(ruta)), artifact_path="procedencia")


def verificar_procedencia(ruta: Path, etapa: str, data_sha256: str) -> dict:
    metadata = _ruta_metadatos(ruta)
    if not ruta.exists() or not metadata.exists():
        raise FileNotFoundError(
            f"Faltan {ruta} o sus metadatos de procedencia. "
            "Vuelve a ejecutar los entry points ingest y features según corresponda."
        )
    registrado = json.loads(metadata.read_text(encoding="utf-8"))
    actual = _metadatos(ruta, etapa, data_sha256)
    if registrado != actual:
        raise ValueError(
            f"Procedencia desactualizada para {ruta}. "
            "Vuelve a ejecutar los entry points ingest y features según corresponda."
        )
    return actual


def verificar_datos_preparados(pregunta: str, data_sha256: str) -> dict:
    if pregunta not in {"p1", "p2", "p3"}:
        raise ValueError(f"Pregunta desconocida: {pregunta}")
    interim = verificar_procedencia(INTERIM, "ingest", data_sha256)
    if pregunta == "p3":
        return {"interim_sha256": interim["archivo_sha256"]}
    features = verificar_procedencia(FEATURES, "features", data_sha256)
    return {
        "interim_sha256": interim["archivo_sha256"],
        "features_sha256": features["archivo_sha256"],
    }
