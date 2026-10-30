import numpy as np

from config import HIPERPARAMS_POR_DEFECTO
from src.train_regresion import crear_modelo_regresion, metricas_regresion


def test_metricas_regresion_caso_normal_y_caso_constante():
    metricas = metricas_regresion([1, 2, 3], [1, 2, 4])
    assert metricas == {"rmse": 0.5773502691896257, "mae": 1 / 3, "r2": 0.5}

    constantes = metricas_regresion([0, 0], [0, 0])
    assert constantes == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}


def test_entrenamiento_p1_es_reproducible_con_la_misma_semilla():
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(80, 3))
    y_train = x_train[:, 0] * 2 - x_train[:, 1] + 0.1
    x_test = rng.normal(size=(20, 3))
    hiperparams = {
        **HIPERPARAMS_POR_DEFECTO["p1"]["histgb"],
        "max_iter": 20,
        "min_samples_leaf": 3,
    }

    primera = crear_modelo_regresion(hiperparams).fit(x_train, y_train).predict(x_test)
    segunda = crear_modelo_regresion(hiperparams).fit(x_train, y_train).predict(x_test)

    np.testing.assert_array_equal(primera, segunda)
