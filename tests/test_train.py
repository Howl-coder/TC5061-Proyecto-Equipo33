from config import HIPERPARAMS_POR_DEFECTO
import pytest

import src.train as modulo_train
from src.train import parsear_hiperparams

PERMITIDOS_P2 = HIPERPARAMS_POR_DEFECTO["p2"]["logreg"]


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("C=0.5,max_iter=2000", {"C": 0.5, "max_iter": 2000}),
        (" C = 2 , max_iter=10 ", {"C": 2, "max_iter": 10}),
        ("class_weight=balanced", {"class_weight": "balanced"}),
        ("class_weight=None", {"class_weight": None}),
        ("C=0.5,,", {"C": 0.5}),
        ("config", {}),
        ("", {}),
    ],
)
def test_parsear_hiperparams_casos_validos(texto, esperado):
    assert parsear_hiperparams(texto, PERMITIDOS_P2) == esperado


@pytest.mark.parametrize("texto", ["gamma=1", "C", "C=1,penalty=l1"])
def test_parsear_hiperparams_rechaza_claves_desconocidas_o_sin_valor(texto):
    with pytest.raises(ValueError, match="Hiperparámetro inválido"):
        parsear_hiperparams(texto, PERMITIDOS_P2)


def test_main_valida_el_arbol_git_antes_de_entrenar_candidato(monkeypatch):
    def rechazar_candidato():
        raise RuntimeError("árbol Git sucio")

    monkeypatch.setattr(
        "sys.argv", ["train", "--pregunta", "p1", "--stage", "candidate"]
    )
    monkeypatch.setattr(modulo_train, "validar_entorno_candidato", rechazar_candidato)
    monkeypatch.setattr(
        modulo_train.train_regresion,
        "cargar_datos",
        lambda: pytest.fail("No debe cargar datos si el árbol Git no está limpio"),
    )

    with pytest.raises(RuntimeError, match="árbol Git sucio"):
        modulo_train.main()
