import pytest

from src import provenance


@pytest.fixture
def datos_preparados(tmp_path, monkeypatch):
    codigo = tmp_path / "src"
    codigo.mkdir()
    (tmp_path / "config.py").write_text("config", encoding="utf-8")
    (codigo / "dataset.py").write_text("dataset", encoding="utf-8")
    (codigo / "features.py").write_text("features", encoding="utf-8")
    interim = tmp_path / "interim.parquet"
    features = tmp_path / "features.parquet"
    interim.write_bytes(b"interim")
    features.write_bytes(b"features")
    monkeypatch.setattr(provenance, "BASE_DIR", tmp_path)
    monkeypatch.setattr(provenance, "INTERIM", interim)
    monkeypatch.setattr(provenance, "FEATURES", features)
    provenance.guardar_procedencia(interim, "ingest", "raw-hash")
    provenance.guardar_procedencia(features, "features", "raw-hash")
    return tmp_path, interim, features


def test_verifica_y_registra_los_hashes_para_cada_pregunta(datos_preparados):
    _, interim, features = datos_preparados

    assert provenance.verificar_datos_preparados("p1", "raw-hash") == {
        "interim_sha256": provenance.sha256_archivo(interim),
        "features_sha256": provenance.sha256_archivo(features),
    }
    assert provenance.verificar_datos_preparados("p2", "raw-hash") == (
        provenance.verificar_datos_preparados("p1", "raw-hash")
    )
    assert provenance.verificar_datos_preparados("p3", "raw-hash") == {
        "interim_sha256": provenance.sha256_archivo(interim)
    }


@pytest.mark.parametrize("cambio", ["raw", "interim", "features", "codigo"])
def test_rechaza_datos_o_codigo_desactualizados(datos_preparados, cambio):
    root, interim, features = datos_preparados
    data_hash = "raw-hash"
    if cambio == "raw":
        data_hash = "nuevo-hash"
    elif cambio == "interim":
        interim.write_bytes(b"otro interim")
    elif cambio == "features":
        features.write_bytes(b"otras features")
    else:
        (root / "src" / "features.py").write_text("nuevo codigo", encoding="utf-8")

    with pytest.raises(ValueError, match="Procedencia desactualizada"):
        provenance.verificar_datos_preparados("p1", data_hash)


def test_requiere_metadatos_y_p3_no_necesita_features(datos_preparados):
    _, _, features = datos_preparados
    provenance._ruta_metadatos(features).unlink()

    provenance.verificar_datos_preparados("p3", "raw-hash")
    with pytest.raises(FileNotFoundError, match="metadatos"):
        provenance.verificar_datos_preparados("p1", "raw-hash")


def test_rechaza_features_anteriores_a_una_nueva_ingesta(datos_preparados):
    _, interim, _ = datos_preparados
    interim.write_bytes(b"nueva ingesta")
    provenance.guardar_procedencia(interim, "ingest", "raw-hash")

    provenance.verificar_datos_preparados("p3", "raw-hash")
    with pytest.raises(ValueError, match="Procedencia desactualizada"):
        provenance.verificar_datos_preparados("p1", "raw-hash")
