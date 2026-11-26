import pandas as pd
import pytest

from src import monitoring_data
from src.monitoring_data import (
    COLUMNAS_DATASET,
    guardar_datasets_monitoreo,
    preparar_datasets_monitoreo,
)


def _datos(tiempos: list[str]) -> pd.DataFrame:
    datos = pd.DataFrame({"timestamp_hora": pd.to_datetime(tiempos), "Origen_Id": 1})
    for columna in COLUMNAS_DATASET[2:]:
        datos[columna] = 1
    return datos


def test_preparar_datasets_separa_referencia_y_septiembre_sin_horas_fuera_del_periodo():
    datos = _datos(
        [
            "2026-04-08 00:00",
            "2026-08-31 23:00",
            "2026-09-01 00:00",
            "2026-09-30 23:00",
            "2026-10-01 00:00",
        ]
    )

    referencia, actual = preparar_datasets_monitoreo(datos)

    assert referencia["timestamp_hora"].tolist() == [
        pd.Timestamp("2026-04-08 00:00"),
        pd.Timestamp("2026-08-31 23:00"),
    ]
    assert actual["timestamp_hora"].tolist() == [
        pd.Timestamp("2026-09-01 00:00"),
        pd.Timestamp("2026-09-30 23:00"),
    ]
    assert referencia.columns.tolist() == COLUMNAS_DATASET
    assert actual.columns.tolist() == COLUMNAS_DATASET


def test_preparar_datasets_rechaza_esquema_incompleto_y_periodo_vacio():
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        preparar_datasets_monitoreo(_datos(["2026-09-01"]).drop(columns=["hora_sin"]))

    with pytest.raises(ValueError, match="conjunto de referencia está vacío"):
        preparar_datasets_monitoreo(_datos(["2026-09-01"]))

    with pytest.raises(ValueError, match="conjunto actual está vacío"):
        preparar_datasets_monitoreo(_datos(["2026-04-08"]))


def test_guardar_datasets_escribe_parquets_y_metadata(tmp_path):
    datos = _datos(["2026-08-31 23:00", "2026-09-01 00:00"])
    referencia, actual = preparar_datasets_monitoreo(datos)

    procedencia = {"data_version": "v1", "data_sha256": "raw", "features_sha256": "features"}
    resumen = guardar_datasets_monitoreo(referencia, actual, tmp_path, procedencia=procedencia)

    assert pd.read_parquet(tmp_path / "referencia_p1.parquet").equals(referencia)
    assert pd.read_parquet(tmp_path / "actual_p1.parquet").equals(actual)
    assert (tmp_path / "metadata.json").exists()
    assert resumen["actual"]["filas"] == 1
    assert resumen["procedencia_fuente"] == procedencia
    assert len(resumen["actual"]["sha256"]) == 64


@pytest.mark.parametrize("cambio", ["fuente", "referencia", "actual"])
def test_verificar_monitoreo_rechaza_fuentes_o_datasets_modificados(tmp_path, monkeypatch, cambio):
    datos = _datos(["2026-08-31 23:00", "2026-09-01 00:00"])
    referencia, actual = preparar_datasets_monitoreo(datos)
    procedencia = {"data_version": "v1", "data_sha256": "raw", "features_sha256": "features"}
    guardar_datasets_monitoreo(referencia, actual, tmp_path, procedencia=procedencia)
    monkeypatch.setattr(monitoring_data, "procedencia_fuente", lambda: procedencia)
    monitoring_data.verificar_datasets_monitoreo(tmp_path)

    if cambio == "fuente":
        monkeypatch.setattr(
            monitoring_data, "procedencia_fuente", lambda: {**procedencia, "data_sha256": "otro"}
        )
    else:
        (tmp_path / f"{cambio}_p1.parquet").write_bytes(b"modificado")

    with pytest.raises(ValueError, match="desactualizad"):
        monitoring_data.verificar_datasets_monitoreo(tmp_path)


def test_verificar_monitoreo_rechaza_metadata_historica_sin_hashes(tmp_path, monkeypatch):
    (tmp_path / "metadata.json").write_text('{"data_version": "v1"}', encoding="utf-8")
    monkeypatch.setattr(monitoring_data, "procedencia_fuente", lambda: {"data_version": "v1"})
    with pytest.raises(ValueError, match="Procedencia de monitoreo desactualizada"):
        monitoring_data.verificar_datasets_monitoreo(tmp_path)
