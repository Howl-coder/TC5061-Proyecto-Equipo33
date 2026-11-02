import numpy as np
import pytest
from sklearn.exceptions import UndefinedMetricWarning

from config import HIPERPARAMS_POR_DEFECTO
from src.train_clasificacion import MODELOS, metricas_clasificacion


def test_metricas_clasificacion_caso_perfecto():
    metricas = metricas_clasificacion(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))

    assert metricas == {"roc_auc": 1.0, "pr_auc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_metricas_clasificacion_umbral_es_inclusivo():
    metricas = metricas_clasificacion(np.array([0, 1]), np.array([0.5, 0.5]))

    assert metricas["recall"] == 1.0
    assert metricas["precision"] == 0.5
    assert metricas["f1"] == pytest.approx(2 / 3)


@pytest.mark.filterwarnings("ignore", category=UndefinedMetricWarning)
def test_metricas_clasificacion_sin_positivos_predichos_no_divide_entre_cero():
    # Ordena bien a las clases pero ninguna probabilidad supera el umbral 0.5
    metricas = metricas_clasificacion(np.array([0, 1, 0, 1]), np.array([0.1, 0.4, 0.2, 0.3]))

    assert metricas["roc_auc"] == 1.0
    assert metricas["precision"] == 0.0
    assert metricas["recall"] == 0.0
    assert metricas["f1"] == 0.0


@pytest.mark.parametrize("nombre_modelo", sorted(MODELOS))
def test_entrenamiento_p2_es_reproducible_con_la_misma_semilla(nombre_modelo):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 5))
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.5, size=400) > 0.8).astype(int)
    _, construir = MODELOS[nombre_modelo]
    hiperparams = dict(HIPERPARAMS_POR_DEFECTO["p2"][nombre_modelo])
    if nombre_modelo == "histgb":
        hiperparams.update(max_iter=30, early_stopping=True, validation_fraction=0.2)

    def entrenar_y_evaluar():
        proba = construir(hiperparams).fit(x[:300], y[:300]).predict_proba(x[300:])[:, 1]
        return proba, metricas_clasificacion(y[300:], proba)

    proba_1, metricas_1 = entrenar_y_evaluar()
    proba_2, metricas_2 = entrenar_y_evaluar()

    np.testing.assert_array_equal(proba_1, proba_2)
    assert metricas_1 == metricas_2
