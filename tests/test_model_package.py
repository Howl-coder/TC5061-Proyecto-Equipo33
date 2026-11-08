import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.model_package import cargar_modelo_empaquetado, exportar_modelo, uri_registro

NOMBRE = "modelo-prueba"


@pytest.mark.parametrize(
    ("version", "alias", "esperado"),
    [
        (3, None, "models:/m/3"),
        ("12", None, "models:/m/12"),
        (None, "champion", "models:/m@champion"),
    ],
)
def test_uri_registro_por_version_o_alias(version, alias, esperado):
    assert uri_registro("m", version=version, alias=alias) == esperado


@pytest.mark.parametrize(
    ("version", "alias"), [(None, None), (1, "champion"), ("latest", None), ("-1", None)]
)
def test_uri_registro_rechaza_referencias_invalidas(version, alias):
    with pytest.raises(ValueError):
        uri_registro("m", version=version, alias=alias)


@pytest.fixture(scope="module")
def registro(tmp_path_factory):
    """Registry SQLite temporal con dos versiones; champion apunta a la primera."""
    base = tmp_path_factory.mktemp("registry")
    tracking_uri = f"sqlite:///{(base / 'mlflow.db').as_posix()}"
    uri_anterior = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experimento = client.create_experiment("prueba", artifact_location=(base / "art").as_uri())

    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
    modelos = {}
    for pendiente in (2.0, 5.0):
        modelo = LinearRegression().fit(X, pendiente * X["a"])
        with mlflow.start_run(experiment_id=experimento) as run:
            mlflow.sklearn.log_model(
                modelo,
                name="modelo",
                signature=infer_signature(X, modelo.predict(X)),
                input_example=X.iloc[:2],
            )
        version = mlflow.register_model(f"runs:/{run.info.run_id}/modelo", NOMBRE).version
        modelos[str(version)] = (modelo, run.info.run_id)
    client.set_registered_model_alias(NOMBRE, "champion", "1")

    yield {"tracking_uri": tracking_uri, "X": X, "modelos": modelos}
    mlflow.set_tracking_uri(uri_anterior)


def test_exportar_por_alias_conserva_linaje_y_predicciones(registro, tmp_path):
    destino = tmp_path / "servicio"
    metadatos = exportar_modelo(
        NOMBRE, destino, alias="champion", tracking_uri=registro["tracking_uri"]
    )
    modelo_original, run_id = registro["modelos"]["1"]

    assert metadatos["version"] == "1"
    assert metadatos["run_id"] == run_id
    assert metadatos["uri_solicitada"] == f"models:/{NOMBRE}@champion"
    assert (destino / "MLmodel").is_file()

    modelo, leidos = cargar_modelo_empaquetado(destino)
    assert leidos == metadatos
    np.testing.assert_allclose(
        modelo.predict(registro["X"]), modelo_original.predict(registro["X"])
    )


def test_exportar_por_version_reemplaza_exportacion_previa(registro, tmp_path):
    destino = tmp_path / "servicio"
    exportar_modelo(NOMBRE, destino, alias="champion", tracking_uri=registro["tracking_uri"])
    metadatos = exportar_modelo(NOMBRE, destino, version=2, tracking_uri=registro["tracking_uri"])

    assert metadatos["version"] == "2"
    _, leidos = cargar_modelo_empaquetado(destino)
    assert leidos["version"] == "2"


def test_exportar_no_sobrescribe_carpetas_ajenas(registro, tmp_path):
    (tmp_path / "otro.txt").write_text("no borrar", encoding="utf-8")
    with pytest.raises(FileExistsError):
        exportar_modelo(NOMBRE, tmp_path, version=1, tracking_uri=registro["tracking_uri"])
    assert (tmp_path / "otro.txt").is_file()


def test_cargar_sin_exportacion_falla(tmp_path):
    with pytest.raises(FileNotFoundError, match="No hay un modelo exportado"):
        cargar_modelo_empaquetado(tmp_path)
