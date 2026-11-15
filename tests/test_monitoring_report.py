import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently")

from src.monitoring_data import COLUMNAS_DATASET
from src.monitoring_report import generar_reporte_evidently
from src.p1_schema import FEATURES


def _datos(inicio: str, cambio: float = 0.0) -> pd.DataFrame:
    n = 80
    datos = pd.DataFrame(
        {
            "timestamp_hora": pd.date_range(inicio, periods=n, freq="h"),
            "Origen_Id": np.resize([1, 2, 3, 4], n),
        }
    )
    for indice, columna in enumerate(FEATURES):
        datos[columna] = np.arange(n, dtype=float) / n + indice + cambio
    datos["demanda_salidas"] = np.arange(n) % 8
    return datos[COLUMNAS_DATASET]


def test_generar_reporte_crea_html_y_json_con_drift_y_calidad(tmp_path):
    referencia = _datos("2026-08-01")
    actual = _datos("2026-09-01", cambio=2.0)
    ruta_html = tmp_path / "drift.html"

    generar_reporte_evidently(referencia, actual, ruta_html)

    contenido = ruta_html.read_text(encoding="utf-8")
    resultado_json = json.loads(ruta_html.with_suffix(".json").read_text(encoding="utf-8"))
    nombres_metricas = [metrica["metric_name"] for metrica in resultado_json["metrics"]]
    assert "Data Drift" in contenido
    assert any(nombre.startswith("ValueDrift(") for nombre in nombres_metricas)
    assert "DatasetMissingValueCount()" in nombres_metricas
    assert "DuplicatedRowCount()" in nombres_metricas


def test_generar_reporte_rechaza_columnas_faltantes(tmp_path):
    datos = _datos("2026-09-01").drop(columns=["lag_salidas_1h"])

    with pytest.raises(ValueError, match="Faltan columnas requeridas en actual"):
        generar_reporte_evidently(_datos("2026-08-01"), datos, tmp_path / "drift.html")
