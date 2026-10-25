"""Utilidades comunes para registrar los runs en MLflow Tracking."""
import subprocess

import mlflow

from config import (
    ARCHIVOS_VIAJES_RAW,
    BASE_DIR,
    EQUIPO,
    FECHA_FIN,
    FECHA_FIN_TRAIN,
    FECHA_INICIO,
    FECHA_INICIO_TEST,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    RUTA_CATALOGO_ESTACIONES,
)
from src.dataset import verificar_datos_crudos

# Modelos HistGradientBoosting entrenados por nosotros; skops exige declarar este tipo
SKOPS_TIPOS_CONFIABLES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=BASE_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def obtener_git_commit() -> str:
    return _git("rev-parse", "HEAD")


def repo_con_cambios() -> bool:
    """True si hay cambios sin commit en archivos versionados."""
    return bool(_git("status", "--porcelain", "--untracked-files=no"))


def configurar_experimento() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def tags_run(pregunta: str, model_family: str, stage: str = "baseline") -> dict:
    return {
        "equipo": EQUIPO,
        "question": pregunta,
        "git_commit": obtener_git_commit(),
        "git_dirty": str(repo_con_cambios()),
        "model_family": model_family,
        "stage_of_work": stage,
    }


def params_reproducibilidad() -> dict:
    """Parámetros comunes; falla si los datos locales no coinciden con el manifiesto versionado."""
    manifiesto = verificar_datos_crudos()
    return {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "fecha_fin_train": FECHA_FIN_TRAIN,
        "fecha_inicio_test": FECHA_INICIO_TEST,
        "data_version": manifiesto["version"],
        "data_sha256": manifiesto["sha256_global"][:16],
        "archivos_viajes": ",".join(p.name for p in ARCHIVOS_VIAJES_RAW),
        "catalogo_estaciones": RUTA_CATALOGO_ESTACIONES.name,
        "random_state": RANDOM_STATE,
    }
