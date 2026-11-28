import json

import pytest

from scripts.validar_modelo_publicacion import version_publicable


@pytest.fixture
def registro():
    return {
        "nombre": "mibici-equipo33-p1-demanda",
        "alias": "champion",
        "version": "1",
        "run_id": "run-1",
        "tags": {
            "source_run_id": "run-1",
            "question": "P1",
            "git_dirty": "False",
            "git_commit": "a" * 40,
            "data_version": "datos-v1",
        },
    }


def test_acepta_modelo_champion_con_linaje(tmp_path, registro):
    ruta = tmp_path / "registro.json"
    ruta.write_text(json.dumps(registro), encoding="utf-8")
    assert version_publicable(ruta) == "1"


@pytest.mark.parametrize("cambio", ["dirty", "run", "alias", "version", "commit", "datos"])
def test_rechaza_linaje_invalido(tmp_path, registro, cambio):
    if cambio == "dirty":
        registro["tags"]["git_dirty"] = "True"
    elif cambio == "run":
        registro["tags"]["source_run_id"] = "otro"
    elif cambio == "alias":
        registro["alias"] = None
    elif cambio == "version":
        registro["version"] = "0"
    elif cambio == "commit":
        registro["tags"]["git_commit"] = "desconocido"
    else:
        del registro["tags"]["data_version"]
    ruta = tmp_path / "registro.json"
    ruta.write_text(json.dumps(registro), encoding="utf-8")
    with pytest.raises(ValueError, match="linaje"):
        version_publicable(ruta)


def test_rechaza_metadatos_sin_estructura(tmp_path):
    ruta = tmp_path / "registro.json"
    ruta.write_text('{"tags": null}', encoding="utf-8")
    with pytest.raises(TypeError, match="linaje"):
        version_publicable(ruta)
