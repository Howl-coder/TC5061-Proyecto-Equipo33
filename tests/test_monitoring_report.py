import json

from docx import Document
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently")

from scripts import generar_reporte_monitoring as documento

from src import monitoring_report
from src.dataset import sha256_archivo
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


def test_reporte_cli_guarda_procedencia_y_hashes(tmp_path, monkeypatch):
    _datos("2026-08-01").to_parquet(tmp_path / "referencia_p1.parquet")
    _datos("2026-09-01").to_parquet(tmp_path / "actual_p1.parquet")
    ruta_html = tmp_path / "drift.html"
    monkeypatch.setattr(monitoring_report, "RUTA_SALIDA", tmp_path)
    monkeypatch.setattr(monitoring_report, "RUTA_REPORTE", ruta_html)
    monkeypatch.setattr(
        monitoring_report, "verificar_datasets_monitoreo", lambda: {"data_version": "v1"}
    )

    monitoring_report.main()

    metadata = json.loads(ruta_html.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["datasets"]["data_version"] == "v1"
    assert metadata["html_sha256"] == sha256_archivo(ruta_html)
    assert metadata["json_sha256"] == sha256_archivo(ruta_html.with_suffix(".json"))


def test_documento_usa_resultados_actuales_y_rechaza_evidencia_mezclada(tmp_path, monkeypatch):
    ruta_html = tmp_path / "drift_p1.html"
    generar_reporte_evidently(_datos("2026-08-01"), _datos("2026-09-01", 2), ruta_html)
    ruta_json = ruta_html.with_suffix(".json")
    ruta_metadata = ruta_html.with_suffix(".metadata.json")
    metadata = {
        "referencia": {"filas": 80, "estaciones": 4, "desde": "2026-08-01", "hasta": "2026-08-04"},
        "actual": {"filas": 80, "estaciones": 4, "desde": "2026-09-01", "hasta": "2026-09-04"},
    }
    ruta_metadata.write_text(
        json.dumps(
            {
                "datasets": metadata,
                "html_sha256": sha256_archivo(ruta_html),
                "json_sha256": sha256_archivo(ruta_json),
            }
        ),
        encoding="utf-8",
    )
    ab = {
        "version_a": "1",
        "version_b": "2",
        "metricas": {
            "1": {"rmse": 2.0, "mae": 1.0, "r2": 0.1},
            "2": {"rmse": 1.0, "mae": 0.5, "r2": 0.5},
        },
        "modelos": {"2": {"hiperparametros": {}, "run_id": "run-2", "git_dirty": "False"}},
        "bootstrap_rmse_b_menos_a": {
            "bloques_dia": 4,
            "replicas": 10,
            "delta_observado_rmse_b_menos_a": -1.0,
            "ic_95_inferior": -1.1,
            "ic_95_superior": -0.9,
        },
        "procedencia_datasets": metadata,
    }
    ruta_ab = tmp_path / "ab.json"
    ruta_ab.write_text(json.dumps(ab), encoding="utf-8")
    monkeypatch.setattr(documento, "RUTA_DRIFT", ruta_json)
    monkeypatch.setattr(documento, "RUTA_DRIFT_METADATA", ruta_metadata)
    monkeypatch.setattr(documento, "RUTA_AB", ruta_ab)
    monkeypatch.setattr(documento, "verificar_datasets_monitoreo", lambda: metadata)
    salida = tmp_path / "reporte.docx"

    documento.generar_documento(salida)

    informe = Document(salida)
    tabla_metricas = next(t for t in informe.tables if t.cell(0, 0).text == "Métrica")
    assert [fila.cells[-1].text for fila in tabla_metricas.rows[1:]] == ["V2", "V2", "V2"]
    texto = "\n".join(p.text for p in informe.paragraphs)
    assert "no modifica el Registry" in texto
    assert "recomienda conservar la versión 1" not in texto

    ab["procedencia_datasets"] = {"otro": "dataset"}
    ruta_ab.write_text(json.dumps(ab), encoding="utf-8")
    with pytest.raises(ValueError, match="mismos datasets"):
        documento.generar_documento(tmp_path / "mezclado.docx")
