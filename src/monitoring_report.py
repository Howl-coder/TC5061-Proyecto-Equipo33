"""Genera un reporte de drift y calidad de datos P1 con Evidently.

Uso:
    python -m src.monitoring_report
"""

import json
from pathlib import Path

from config import REPORTS_DIR
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import pandas as pd

from src.dataset import sha256_archivo
from src.monitoring_data import COLUMNAS_DATASET, RUTA_SALIDA, verificar_datasets_monitoreo
from src.p1_schema import FEATURES, TARGET

RUTA_REPORTE = REPORTS_DIR / "monitoring" / "drift_p1.html"
COLUMNAS_DRIFT = [*FEATURES, TARGET]


def generar_reporte_evidently(
    referencia: pd.DataFrame, actual: pd.DataFrame, ruta_html: Path = RUTA_REPORTE
):
    """Compara drift de variables P1 y calidad de datos y guarda el reporte HTML."""
    for nombre, datos in (("referencia", referencia), ("actual", actual)):
        faltantes = [columna for columna in COLUMNAS_DATASET if columna not in datos.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas requeridas en {nombre}: {faltantes}")
        if datos.empty:
            raise ValueError(f"El conjunto {nombre} está vacío.")

    reporte = Report(
        [
            DataDriftPreset(columns=COLUMNAS_DRIFT),
            DataSummaryPreset(),
        ]
    )
    resultado = reporte.run(current_data=actual, reference_data=referencia)

    ruta_html.parent.mkdir(parents=True, exist_ok=True)
    resultado.save_html(str(ruta_html))
    ruta_html.with_suffix(".json").write_text(resultado.json(), encoding="utf-8")
    return resultado


def main() -> None:
    ruta_referencia = RUTA_SALIDA / "referencia_p1.parquet"
    ruta_actual = RUTA_SALIDA / "actual_p1.parquet"
    faltantes = [ruta for ruta in (ruta_referencia, ruta_actual) if not ruta.exists()]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan los datasets de monitoreo: {faltantes}. "
            "Ejecuta primero `python -m src.monitoring_data`."
        )

    procedencia = verificar_datasets_monitoreo()
    referencia = pd.read_parquet(ruta_referencia)
    actual = pd.read_parquet(ruta_actual)
    generar_reporte_evidently(referencia, actual, ruta_html=RUTA_REPORTE)
    metadatos = {
        "datasets": procedencia,
        "html_sha256": sha256_archivo(RUTA_REPORTE),
        "json_sha256": sha256_archivo(RUTA_REPORTE.with_suffix(".json")),
    }
    ruta_metadata = RUTA_REPORTE.with_suffix(".metadata.json")
    ruta_metadata.write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Reporte Evidently guardado en {RUTA_REPORTE}")
    print(f"Resultados estructurados guardados en {RUTA_REPORTE.with_suffix('.json')}")
    print(f"Procedencia del reporte guardada en {ruta_metadata}")


if __name__ == "__main__":
    main()
