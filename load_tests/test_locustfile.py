from unittest.mock import MagicMock

from load_tests.locustfile import UsuarioPrediccion
from locust.env import Environment
import pytest


@pytest.fixture
def usuario(monkeypatch):
    monkeypatch.setattr(UsuarioPrediccion, "host", "http://localhost")
    return UsuarioPrediccion(Environment())


@pytest.mark.parametrize(
    ("status", "cuerpo", "fallo"),
    [
        (201, {}, "HTTP 201"),
        (500, {}, "HTTP 500"),
        (
            200,
            {"target": "demanda_salidas", "modelo": "p1", "version": "1", "predicciones": []},
            "La respuesta no cumple el contrato de predicción",
        ),
    ],
)
def test_solicitud_reporta_respuestas_invalidas(usuario, status, cuerpo, fallo):
    respuesta = MagicMock(status_code=status)
    respuesta.json.return_value = cuerpo
    usuario.client.post = MagicMock()
    usuario.client.post.return_value.__enter__.return_value = respuesta

    usuario.predecir()

    respuesta.failure.assert_called_once_with(fallo)


def test_solicitud_acepta_prediccion_valida(usuario):
    respuesta = MagicMock(status_code=200)
    respuesta.json.return_value = {
        "target": "demanda_salidas",
        "modelo": "p1",
        "version": "1",
        "predicciones": [3.0],
    }
    usuario.client.post = MagicMock()
    usuario.client.post.return_value.__enter__.return_value = respuesta

    usuario.predecir()

    respuesta.failure.assert_not_called()
