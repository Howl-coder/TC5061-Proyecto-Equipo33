"""Prepara los datasets de referencia y actual para monitoreo de P1.

Uso:
    python -m src.monitoring_data

Parte de ``data/processed/features_demanda_p1.parquet``, generado con las mismas
transformaciones usadas por el entrenamiento, y escribe los conjuntos y un resumen
en ``data/monitoring/``.
"""

import json
from pathlib import Path

from config import (
    DATA_DIR,
    DATA_VERSION,
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_TEST,
    PROCESSED_DATA_DIR,
)
import pandas as pd

from src.p1_schema import FEATURES, TARGET

COLUMNA_TIEMPO = "timestamp_hora"
COLUMNAS_DATASET = [COLUMNA_TIEMPO, "Origen_Id", *FEATURES, TARGET]
RUTA_FEATURES = PROCESSED_DATA_DIR / "features_demanda_p1.parquet"
RUTA_SALIDA = DATA_DIR / "monitoring"


def preparar_datasets_monitoreo(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa las filas P1 en referencia de entrenamiento y periodo actual posterior."""
    faltantes = [columna for columna in COLUMNAS_DATASET if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas para monitoreo: {faltantes}")
    if not pd.api.types.is_datetime64_any_dtype(datos[COLUMNA_TIEMPO]):
        raise ValueError(f"'{COLUMNA_TIEMPO}' debe tener tipo datetime64.")

    inicio_referencia = pd.Timestamp(FECHA_INICIO)
    inicio_actual = pd.Timestamp(FECHA_INICIO_TEST)
    fin_actual_exclusivo = pd.Timestamp(FECHA_FIN).normalize() + pd.Timedelta(days=1)
    periodo = datos[COLUMNA_TIEMPO]
    columnas = COLUMNAS_DATASET

    referencia = datos.loc[
        periodo.ge(inicio_referencia) & periodo.lt(inicio_actual), columnas
    ].copy()
    actual = datos.loc[
        periodo.ge(inicio_actual) & periodo.lt(fin_actual_exclusivo), columnas
    ].copy()

    if referencia.empty:
        raise ValueError("El conjunto de referencia está vacío.")
    if actual.empty:
        raise ValueError("El conjunto actual está vacío.")
    return referencia.reset_index(drop=True), actual.reset_index(drop=True)


def _resumen_periodo(datos: pd.DataFrame, archivo: str) -> dict:
    return {
        "archivo": archivo,
        "filas": len(datos),
        "estaciones": int(datos["Origen_Id"].nunique()),
        "desde": datos[COLUMNA_TIEMPO].min().isoformat(),
        "hasta": datos[COLUMNA_TIEMPO].max().isoformat(),
    }


def guardar_datasets_monitoreo(
    referencia: pd.DataFrame, actual: pd.DataFrame, directorio: Path = RUTA_SALIDA
) -> dict:
    """Guarda ambos conjuntos como Parquet y un resumen reproducible en JSON."""
    salida = Path(directorio)
    salida.mkdir(parents=True, exist_ok=True)

    archivo_referencia = "referencia_p1.parquet"
    archivo_actual = "actual_p1.parquet"
    referencia.to_parquet(salida / archivo_referencia, index=False)
    actual.to_parquet(salida / archivo_actual, index=False)

    resumen = {
        "data_version": DATA_VERSION,
        "fuente": str(RUTA_FEATURES.relative_to(DATA_DIR.parent)),
        "target": TARGET,
        "features": FEATURES,
        "referencia": {
            "periodo_configurado": f"{FECHA_INICIO} a {FECHA_INICIO_TEST} (exclusivo)",
            **_resumen_periodo(referencia, archivo_referencia),
        },
        "actual": {
            "periodo_configurado": f"{FECHA_INICIO_TEST} a {FECHA_FIN}",
            **_resumen_periodo(actual, archivo_actual),
        },
    }
    with open(salida / "metadata.json", "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=2, ensure_ascii=False)
        archivo.write("\n")
    return resumen


def main() -> None:
    if not RUTA_FEATURES.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_FEATURES}. Ejecuta primero `python -m src.dataset` "
            "y `python -m src.features`."
        )

    datos = pd.read_parquet(RUTA_FEATURES)
    referencia, actual = preparar_datasets_monitoreo(datos)
    resumen = guardar_datasets_monitoreo(referencia, actual)
    for conjunto in ("referencia", "actual"):
        periodo = resumen[conjunto]
        print(
            f"{conjunto}: {periodo['filas']:,} filas, {periodo['estaciones']} estaciones, "
            f"{periodo['desde']} a {periodo['hasta']} "
            f"({RUTA_SALIDA / periodo['archivo']})"
        )


if __name__ == "__main__":
    main()
