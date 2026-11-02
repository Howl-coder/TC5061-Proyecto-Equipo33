import hashlib

import pandas as pd
import pytest

from config import DATA_VERSION
from src.dataset import cargar_catalogo_amg, generar_manifiesto, sha256_archivo


@pytest.mark.parametrize(
    ("contenido", "esperado"),
    [
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ],
)
def test_sha256_archivo_coincide_con_hash_conocido(tmp_path, contenido, esperado):
    ruta = tmp_path / "archivo.bin"
    ruta.write_bytes(contenido)

    assert sha256_archivo(ruta) == esperado


def test_sha256_archivo_lee_por_bloques_archivos_grandes(tmp_path):
    # Más de un bloque de 1 MB para ejercitar el ciclo de lectura
    contenido = b"x" * ((1 << 20) * 2 + 7)
    ruta = tmp_path / "grande.bin"
    ruta.write_bytes(contenido)

    assert sha256_archivo(ruta) == hashlib.sha256(contenido).hexdigest()


def test_generar_manifiesto_describe_archivos_y_es_determinista(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("id\n1\n", encoding="utf-8")
    b.write_text("id\n2\n3\n", encoding="utf-8")

    manifiesto = generar_manifiesto([a, b])

    assert manifiesto["version"] == DATA_VERSION
    assert [e["archivo"] for e in manifiesto["archivos"]] == ["a.csv", "b.csv"]
    assert manifiesto["archivos"][0]["bytes"] == a.stat().st_size
    assert manifiesto["archivos"][1]["sha256"] == sha256_archivo(b)
    assert generar_manifiesto([a, b]) == manifiesto


def test_generar_manifiesto_detecta_cambios_en_los_datos(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("id\n1\n", encoding="utf-8")
    original = generar_manifiesto([a])["sha256_global"]

    a.write_text("id\n1\n2\n", encoding="utf-8")

    assert generar_manifiesto([a])["sha256_global"] != original


def test_generar_manifiesto_falla_si_falta_un_archivo(tmp_path):
    existente = tmp_path / "a.csv"
    existente.write_text("id\n1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="no_existe.csv"):
        generar_manifiesto([existente, tmp_path / "no_existe.csv"])


def _escribir_catalogo(ruta, encoding="utf-8"):
    pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "name": ["Centro", "Límite AMG", "Fuera lat", "Fuera lon"],
            "latitude": [20.67, 20.50, 21.00, 20.67],
            "longitude": [-103.35, -103.50, -103.35, -104.00],
            "dpcapacity": [15, 10, 12, 8],
        }
    ).to_csv(ruta, index=False, encoding=encoding)


def test_cargar_catalogo_amg_filtra_por_bounding_box_incluyendo_bordes(tmp_path):
    ruta = tmp_path / "nomenclatura.csv"
    _escribir_catalogo(ruta)

    catalogo = cargar_catalogo_amg(ruta)

    assert catalogo["id"].tolist() == [1, 2]
    assert pd.api.types.is_integer_dtype(catalogo["id"])


def test_cargar_catalogo_amg_recurre_a_latin1_si_no_es_utf8(tmp_path):
    ruta = tmp_path / "nomenclatura_latin1.csv"
    _escribir_catalogo(ruta, encoding="latin-1")

    catalogo = cargar_catalogo_amg(ruta)

    assert catalogo.loc[catalogo["id"] == 2, "name"].item() == "Límite AMG"
