import json
import math

from fastapi.testclient import TestClient
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
from sklearn.linear_model import LinearRegression

from src.api import EntradaDemanda, crear_app
from src.model_package import ARCHIVO_METADATOS
from src.p1_schema import FEATURES

INSTANCIA = EntradaDemanda.model_config["json_schema_extra"]["examples"][0]


def _calendario(hora: float, dia_semana: float) -> dict:
    return {
        "hora_sin": math.sin(2 * math.pi * hora / 24),
        "hora_cos": math.cos(2 * math.pi * hora / 24),
        "dia_sin": math.sin(2 * math.pi * dia_semana / 7),
        "dia_cos": math.cos(2 * math.pi * dia_semana / 7),
    }


def _guardar_modelo(ruta, X, y):
    modelo = LinearRegression().fit(X, y)
    mlflow.sklearn.save_model(
        modelo, str(ruta), signature=infer_signature(X, modelo.predict(X)), input_example=X[:2]
    )
    metadatos = {"nombre": "mibici-prueba-p1", "version": "7", "alias": "champion", "run_id": "r1"}
    (ruta / ARCHIVO_METADATOS).write_text(json.dumps(metadatos), encoding="utf-8")


@pytest.fixture(scope="module")
def ruta_modelo(tmp_path_factory):
    """Modelo P1 de juguete: predice lag_salidas_1h - 1."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 10, size=(50, len(FEATURES))), columns=FEATURES)
    X["es_fin_de_semana"] = rng.integers(0, 2, size=50)
    ruta = tmp_path_factory.mktemp("export") / "servicio"
    _guardar_modelo(ruta, X, X["lag_salidas_1h"] - 1)
    return ruta


@pytest.fixture(scope="module")
def cliente(ruta_modelo):
    with TestClient(crear_app(ruta_modelo)) as cliente:
        yield cliente


def test_esquema_de_entrada_coincide_con_features_p1():
    assert list(EntradaDemanda.model_fields) == FEATURES


def test_health_y_metadatos(cliente):
    assert cliente.get("/health").json() == {"status": "ok"}
    info = cliente.get("/modelo").json()
    assert info["nombre"] == "mibici-prueba-p1"
    assert info["version"] == "7"
    assert info["features"] == FEATURES


def test_predict_devuelve_json_y_recorta_negativos(cliente):
    instancias = [{**INSTANCIA, "lag_salidas_1h": 5}, {**INSTANCIA, "lag_salidas_1h": 0}]
    respuesta = cliente.post("/predict", json={"instancias": instancias})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["modelo"] == "mibici-prueba-p1"
    assert cuerpo["target"] == "demanda_salidas"
    np.testing.assert_allclose(cuerpo["predicciones"], [4.0, 0.0], atol=1e-6)


@pytest.mark.parametrize(
    "instancias",
    [
        [],
        [{k: v for k, v in INSTANCIA.items() if k != "lag_salidas_1h"}],
        [{**INSTANCIA, "Origen_Id": 10}],
        [{**INSTANCIA, "hora_sin": 2}],
        [{**INSTANCIA, "lag_salidas_24h": -1}],
        [{**INSTANCIA, "es_fin_de_semana": 3}],
        [{**INSTANCIA, "rolling_mean_6h": "mucho"}],
        # sin² + cos² ≠ 1
        [{**INSTANCIA, "hora_sin": 0.5, "hora_cos": 0.5}],
        # 8:30 no es una hora entera de la malla horaria
        [{**INSTANCIA, **_calendario(8.5, 0)}],
        # Sábado marcado como día hábil
        [{**INSTANCIA, **_calendario(8, 5), "es_fin_de_semana": 0}],
    ],
)
def test_predict_rechaza_entradas_invalidas(cliente, instancias):
    assert cliente.post("/predict", json={"instancias": instancias}).status_code == 422


@pytest.mark.parametrize("valor", ["NaN", "Infinity"])
def test_predict_rechaza_valores_no_finitos(cliente, valor):
    # json.dumps emite NaN/Infinity, que Python acepta al decodificar
    cuerpo = json.dumps({"instancias": [{**INSTANCIA, "lag_salidas_1h": float(valor)}]})
    respuesta = cliente.post(
        "/predict", content=cuerpo, headers={"Content-Type": "application/json"}
    )
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"] == ["body", "instancias", 0, "lag_salidas_1h"]


def test_error_422_explica_la_incoherencia(cliente):
    instancia = {**INSTANCIA, "es_fin_de_semana": 1}
    detalle = cliente.post("/predict", json={"instancias": [instancia]}).json()["detail"]
    assert detalle == [
        {
            "type": "value_error",
            "loc": ["body", "instancias", 0],
            "msg": "Value error, es_fin_de_semana no coincide con el día de dia_sin/dia_cos.",
        }
    ]


@pytest.mark.parametrize(
    ("hora", "dia_semana", "fin_de_semana"),
    [(0, 0, 0), (23, 4, 0), (8, 5, 1), (17, 6, 1)],
)
def test_entrada_acepta_calendario_coherente(hora, dia_semana, fin_de_semana):
    datos = {**INSTANCIA, **_calendario(hora, dia_semana), "es_fin_de_semana": fin_de_semana}
    assert EntradaDemanda(**datos).es_fin_de_semana == fin_de_semana


def test_entrada_acepta_senos_redondeados():
    datos = {k: round(v, 3) for k, v in _calendario(13, 2).items()}
    EntradaDemanda(**{**INSTANCIA, **datos})


def test_entrada_informa_campo_faltante():
    with pytest.raises(ValidationError, match="lag_salidas_3h"):
        EntradaDemanda(**{k: v for k, v in INSTANCIA.items() if k != "lag_salidas_3h"})


def test_rechaza_modelo_con_otras_features(tmp_path):
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 0.0, 1.0]})
    ruta = tmp_path / "servicio"
    _guardar_modelo(ruta, X, X["a"])
    with pytest.raises(ValueError, match="la API sirve P1"), TestClient(crear_app(ruta)):
        pass
