"""Entrena un baseline (un run de MLflow) para la pregunta y el modelo indicados.

Uso:
    python -m src.train --pregunta p2 --modelo logreg --hiperparams "C=0.5,max_iter=2000"

Si ``--modelo`` o ``--hiperparams`` se omiten o valen ``config``, se usan los valores de
``config.py``; los hiperparámetros indicados solo reemplazan a los del modelo por defecto.
"""
import argparse
import ast

from config import HIPERPARAMS_POR_DEFECTO, MODELOS_POR_DEFECTO
from src import train_clasificacion, train_clustering, train_regresion
from src.tracking import configurar_experimento, params_reproducibilidad

MODULOS = {"p1": train_regresion, "p2": train_clasificacion, "p3": train_clustering}
# Centinela sin comillas: MLflow en Windows no transmite bien un string vacío
USAR_CONFIG = "config"


def parsear_hiperparams(texto: str, permitidos: dict) -> dict:
    """Convierte ``"clave=valor,clave=valor"`` en dict; los valores se leen como literales de Python."""
    if texto == USAR_CONFIG:
        return {}
    hiperparams = {}
    for par in filter(None, map(str.strip, texto.split(","))):
        clave, signo, valor = par.partition("=")
        clave, valor = clave.strip(), valor.strip()
        if not signo or clave not in permitidos:
            raise ValueError(
                f"Hiperparámetro inválido '{par}'. Usa clave=valor con claves en {sorted(permitidos)}"
            )
        try:
            hiperparams[clave] = ast.literal_eval(valor)
        except (ValueError, SyntaxError):
            hiperparams[clave] = valor
    return hiperparams


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pregunta", required=True, choices=sorted(MODULOS))
    parser.add_argument("--modelo", default=USAR_CONFIG, help="Por defecto, config.MODELOS_POR_DEFECTO.")
    parser.add_argument("--hiperparams", default=USAR_CONFIG, help="clave=valor separados por comas.")
    args = parser.parse_args()

    modelo = MODELOS_POR_DEFECTO[args.pregunta] if args.modelo == USAR_CONFIG else args.modelo
    defaults = HIPERPARAMS_POR_DEFECTO[args.pregunta]
    if modelo not in defaults:
        parser.error(f"Modelo '{modelo}' no válido para {args.pregunta}. Opciones: {sorted(defaults)}")
    try:
        hiperparams = {**defaults[modelo], **parsear_hiperparams(args.hiperparams, defaults[modelo])}
    except ValueError as e:
        parser.error(str(e))

    modulo = MODULOS[args.pregunta]
    datos = modulo.cargar_datos()
    configurar_experimento()
    modulo.entrenar(modelo, hiperparams, datos, params_reproducibilidad())


if __name__ == "__main__":
    main()
