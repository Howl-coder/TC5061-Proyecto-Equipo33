import math

import pytest
from scripts.graficar_carga import leer_historial


def test_historial_calcula_tasa_acumulada(tmp_path):
    ruta = tmp_path / "historial.csv"
    ruta.write_text(
        "Timestamp,Name,Total Request Count,Total Failure Count,50%,95%,Requests/s\n"
        "100,Aggregated,0,0,N/A,N/A,0\n"
        "101,Aggregated,10,2,12,25,5\n",
        encoding="utf-8",
    )
    datos = leer_historial(ruta)
    assert len(datos) == 2
    assert math.isnan(datos[0]["errores"])
    assert math.isnan(datos[0]["p50"])
    assert datos[1] == {"segundos": 1, "p50": 12, "p95": 25, "rps": 5, "errores": 20}


def test_historial_rechaza_full_history(tmp_path):
    ruta = tmp_path / "historial.csv"
    ruta.write_text("Timestamp,Name\n100,/predict\n100,Aggregated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sin --csv-full-history"):
        leer_historial(ruta)


def test_historial_rechaza_conteos_invalidos(tmp_path):
    ruta = tmp_path / "historial.csv"
    ruta.write_text(
        "Timestamp,Name,Total Request Count,Total Failure Count,50%,95%,Requests/s\n"
        "100,Aggregated,1,2,10,20,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conteos"):
        leer_historial(ruta)
