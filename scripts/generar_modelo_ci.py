"""Genera un modelo sintético P1 para comprobar la imagen en CI, no para producción."""

import argparse
import json
from pathlib import Path

from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.model_package import ARCHIVO_METADATOS
from src.p1_schema import FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destino", type=Path, help="Directorio nuevo para el modelo sintético")
    destino = parser.parse_args().destino

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 10, size=(50, len(FEATURES))), columns=FEATURES)
    X["es_fin_de_semana"] = rng.integers(0, 2, size=len(X))
    modelo = LinearRegression().fit(X, X["lag_salidas_1h"])
    mlflow.sklearn.save_model(
        modelo,
        str(destino),
        serialization_format="skops",
        signature=infer_signature(X, modelo.predict(X)),
        input_example=X.iloc[:2],
    )
    metadatos = {"nombre": "mibici-prueba-ci", "version": "1", "alias": None, "run_id": "ci"}
    (destino / ARCHIVO_METADATOS).write_text(
        json.dumps(metadatos, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
