import os
from pathlib import Path

# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Lista EXPLÍCITA de archivos mensuales (sin glob)
ARCHIVOS_VIAJES_RAW = [
    RAW_DATA_DIR / "datos_abiertos_2026_04.csv",
    RAW_DATA_DIR / "datos_abiertos_2026_05.csv",
    RAW_DATA_DIR / "datos_abiertos_2026_06.csv",
    RAW_DATA_DIR / "datos_abiertos_2026_07.csv",
    RAW_DATA_DIR / "datos_abiertos_2026_08.csv",
    RAW_DATA_DIR / "datos_abiertos_2026_09.csv",
]

# Archivo de catálogo de estaciones
RUTA_CATALOGO_ESTACIONES = RAW_DATA_DIR / "nomenclatura_2026_09.csv"

# URLs de descarga del portal de datos abiertos de MiBici
URLS_VIAJES_RAW = {
    "datos_abiertos_2026_04.csv": "https://www.mibici.net/site/assets/files/124927/datos_abiertos_2026_04.csv",
    "datos_abiertos_2026_05.csv": "https://www.mibici.net/site/assets/files/125943/datos_abiertos_2026_05.csv",
    "datos_abiertos_2026_06.csv": "https://www.mibici.net/site/assets/files/127734/datos_abiertos_2026_06.csv",
    "datos_abiertos_2026_07.csv": "https://www.mibici.net/site/assets/files/129006/datos_abiertos_2026_07.csv",
    "datos_abiertos_2026_08.csv": "https://www.mibici.net/site/assets/files/130362/datos_abiertos_2026_08.csv",
    "datos_abiertos_2026_09.csv": "https://www.mibici.net/site/assets/files/131825/datos_abiertos_2026_09.csv",
}
URL_CATALOGO_ESTACIONES = "https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"

# Manifiesto versionado en Git con el SHA-256 de cada archivo crudo
RUTA_MANIFIESTO_DATOS = BASE_DIR / "references" / "manifiesto_datos.json"

# Parámetros temporales y de calidad
FECHA_INICIO = "2026-04-01 00:00:00"
FECHA_FIN = "2026-09-30 23:59:59"
DURACION_MIN_MINUTOS = 1.0
DURACION_MAX_MINUTOS = 1440.0

# Bounding Box: Área Metropolitana de Guadalajara (AMG)
LAT_MIN, LAT_MAX = 20.50, 20.80
LON_MIN, LON_MAX = -103.50, -103.20

# Partición temporal (Train: abr–ago, Test: sep)
FECHA_FIN_TRAIN = "2026-08-31 23:59:59"
FECHA_INICIO_TEST = "2026-09-01 00:00:00"

# MLOps
RANDOM_STATE = 42
EQUIPO = "equipo33"
DATA_VERSION = "mibici-datos-abiertos-2026_04-2026_09"
# Servidor local respaldado por mlflow.db (necesario para Model Registry)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "mibici-equipo33"
FIGURES_DIR = REPORTS_DIR / "figures"

# Baselines por pregunta (defaults del entry point `train` del MLproject); la semilla es RANDOM_STATE
MODELOS_POR_DEFECTO = {"p1": "histgb", "p2": "histgb", "p3": "kmeans"}
HIPERPARAMS_POR_DEFECTO = {
    "p1": {
        "naive": {"columna_naive": "lag_salidas_168h"},
        "histgb": {"max_iter": 200, "learning_rate": 0.1, "max_depth": None},
    },
    "p2": {
        "logreg": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
        "histgb": {
            "max_iter": 200,
            "learning_rate": 0.1,
            "max_depth": None,
            "class_weight": "balanced",
        },
    },
    "p3": {
        "kmeans": {"k_min": 3, "k_max": 10, "n_init": 10},
    },
}