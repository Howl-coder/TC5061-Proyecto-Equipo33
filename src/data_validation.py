"""Validación de integridad y drift de los features temporales con DeepChecks.

Uso:
    python -m src.data_validation
"""

from config import FECHA_INICIO_TEST, PROCESSED_DATA_DIR, REPORTS_DIR
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation
import pandas as pd

from src.p1_schema import FEATURES, TARGET


def validar_datasets(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Ejecuta checks de integridad sobre train y de drift entre train y test."""
    if train.empty or test.empty:
        raise ValueError("Los conjuntos de entrenamiento y prueba deben contener filas.")
    if TARGET not in train or TARGET not in test:
        raise ValueError(f"Ambos conjuntos deben incluir el target '{TARGET}'.")

    columnas = ["Origen_Id", *FEATURES, TARGET]
    requeridas = ["timestamp_hora", *columnas]
    faltantes = [col for col in requeridas if col not in train or col not in test]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas para validar: {faltantes}")

    cat_features = ["Origen_Id"] if "Origen_Id" in columnas else []
    integrity_dataset = Dataset(
        train[requeridas],
        label=TARGET,
        cat_features=cat_features,
        datetime_name="timestamp_hora",
    )
    train_dataset = Dataset(train[columnas], label=TARGET, cat_features=cat_features)
    test_dataset = Dataset(test[columnas], label=TARGET, cat_features=cat_features)

    return {
        "integridad": data_integrity().run(integrity_dataset),
        "train_test": train_test_validation().run(train_dataset, test_dataset),
    }


def main() -> None:
    ruta = PROCESSED_DATA_DIR / "features_demanda_p1.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero `python -m src.dataset` y "
            "`python -m src.features`."
        )

    datos = pd.read_parquet(ruta)
    if "timestamp_hora" not in datos:
        raise ValueError("El dataset debe incluir la columna 'timestamp_hora'.")

    corte = pd.Timestamp(FECHA_INICIO_TEST)
    train = datos.loc[datos["timestamp_hora"] < corte]
    test = datos.loc[datos["timestamp_hora"] >= corte]
    resultados = validar_datasets(train, test)

    salida = REPORTS_DIR / "data-validation"
    salida.mkdir(parents=True, exist_ok=True)
    for nombre, resultado in resultados.items():
        reporte = salida / f"{nombre}.html"
        if reporte.exists():
            reporte.unlink()
        resultado.save_as_html(str(reporte))
        hallazgos = [
            str(item).split(":", 1)[0].removeprefix("CheckResult ")
            for item in resultado.get_not_passed_checks()
        ]
        print(
            f"{nombre}: {len(hallazgos)} hallazgo(s): "
            f"{', '.join(hallazgos) if hallazgos else 'ninguno'}. "
            f"Reporte: {reporte}"
        )
    print("DeepChecks validation completed.")


if __name__ == "__main__":
    main()
