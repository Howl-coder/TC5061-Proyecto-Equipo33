"""Validación de integridad, calidad y drift de los features temporales P1 con DeepChecks.

Uso:
    python -m src.data_validation

Se ejecutan tres suites y se guarda un informe HTML por cada una en reports/data-validation/:
    - integridad (bloqueante): tipos mixtos, nulos, duplicados y columnas constantes,
      evaluados por separado en train y test.
    - particion (bloqueante): tamaño de los conjuntos y fuga temporal entre train y test.
    - drift (advertencia): drift por variable, del target y multivariado, y estaciones nuevas.
El proceso termina con código 1 si el esquema o alguna condición bloqueante falla.
"""

import sys

from config import FECHA_INICIO_TEST, PROCESSED_DATA_DIR, REPORTS_DIR
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    DataDuplicates,
    DatasetsSizeComparison,
    DateTrainTestLeakageOverlap,
    FeatureDrift,
    IsSingleValue,
    LabelDrift,
    MixedDataTypes,
    MixedNulls,
    MultivariateDrift,
    NewCategoryTrainTest,
    PercentOfNulls,
)
import pandas as pd

from src.p1_schema import FEATURES, TARGET

COLUMNA_TIEMPO = "timestamp_hora"
COLUMNA_ESTACION = "Origen_Id"
COLUMNAS = [COLUMNA_TIEMPO, COLUMNA_ESTACION, *FEATURES, TARGET]
SUITES_BLOQUEANTES = ("integridad", "particion")
# Evita que DeepChecks muestree: los conteos de nulos y duplicados deben ser exhaustivos
N_MUESTRAS = 10_000_000


def validar_esquema(df: pd.DataFrame) -> list[str]:
    """Verifica columnas, tipos esperados, rango del target y unicidad de la llave estación-hora."""
    faltantes = [col for col in COLUMNAS if col not in df]
    if faltantes:
        return [f"Faltan columnas requeridas: {faltantes}"]

    problemas = []
    if not pd.api.types.is_datetime64_any_dtype(df[COLUMNA_TIEMPO]):
        problemas.append(f"'{COLUMNA_TIEMPO}' debe ser datetime64, es {df[COLUMNA_TIEMPO].dtype}")
    for col in (COLUMNA_ESTACION, TARGET):
        if not pd.api.types.is_integer_dtype(df[col]):
            problemas.append(f"'{col}' debe ser entero, es {df[col].dtype}")
    no_numericas = [col for col in FEATURES if not pd.api.types.is_numeric_dtype(df[col])]
    if no_numericas:
        problemas.append(f"Features no numéricas: {no_numericas}")
    if pd.api.types.is_numeric_dtype(df[TARGET]) and (df[TARGET] < 0).any():
        problemas.append(f"'{TARGET}' tiene valores negativos")
    n_repetidas = int(df.duplicated([COLUMNA_ESTACION, COLUMNA_TIEMPO]).sum())
    if n_repetidas:
        problemas.append(f"{n_repetidas} fila(s) repiten la llave ({COLUMNA_ESTACION}, {COLUMNA_TIEMPO})")
    return problemas


def suite_integridad() -> Suite:
    return Suite(
        "Integridad y calidad de datos",
        MixedDataTypes(n_samples=N_MUESTRAS).add_condition_rare_type_ratio_not_in_range(),
        PercentOfNulls(n_samples=N_MUESTRAS).add_condition_percent_of_nulls_not_greater_than(0),
        MixedNulls(n_samples=N_MUESTRAS).add_condition_different_nulls_less_equal_to(1),
        DataDuplicates(n_samples=N_MUESTRAS).add_condition_ratio_less_or_equal(0),
        IsSingleValue(n_samples=N_MUESTRAS).add_condition_not_single_value(),
    )


def suite_particion() -> Suite:
    return Suite(
        "Partición temporal train/test",
        DatasetsSizeComparison()
        .add_condition_test_size_greater_or_equal(100)
        .add_condition_train_dataset_greater_or_equal_test(),
        DateTrainTestLeakageOverlap(n_samples=N_MUESTRAS).add_condition_leakage_ratio_less_or_equal(0),
    )


def suite_drift() -> Suite:
    return Suite(
        "Drift entre train y test",
        FeatureDrift().add_condition_drift_score_less_than(),
        LabelDrift().add_condition_drift_score_less_than(),
        MultivariateDrift().add_condition_overall_drift_value_less_than(),
        NewCategoryTrainTest().add_condition_new_category_ratio_less_or_equal(0),
    )


def _dataset(df: pd.DataFrame) -> Dataset:
    return Dataset(
        df[COLUMNAS],
        label=TARGET,
        cat_features=[COLUMNA_ESTACION],
        datetime_name=COLUMNA_TIEMPO,
    )


def validar_datasets(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Valida el esquema y ejecuta las suites de integridad, partición y drift."""
    if train.empty or test.empty:
        raise ValueError("Los conjuntos de entrenamiento y prueba deben contener filas.")
    for nombre, df in (("train", train), ("test", test)):
        problemas = validar_esquema(df)
        if problemas:
            raise ValueError(f"Esquema inválido en {nombre}: {'; '.join(problemas)}")

    train_ds, test_ds = _dataset(train), _dataset(test)
    return {
        "integridad": suite_integridad().run(train_dataset=train_ds, test_dataset=test_ds),
        "particion": suite_particion().run(train_dataset=train_ds, test_dataset=test_ds),
        "drift": suite_drift().run(train_dataset=train_ds, test_dataset=test_ds),
    }


def checks_fallidos(resultado) -> list[str]:
    return [check.get_header() for check in resultado.get_not_passed_checks()]


def main() -> int:
    ruta = PROCESSED_DATA_DIR / "features_demanda_p1.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero `python -m src.dataset` y "
            "`python -m src.features`."
        )

    datos = pd.read_parquet(ruta)
    if COLUMNA_TIEMPO not in datos:
        raise ValueError(f"El dataset debe incluir la columna '{COLUMNA_TIEMPO}'.")

    corte = pd.Timestamp(FECHA_INICIO_TEST)
    train = datos.loc[datos[COLUMNA_TIEMPO] < corte]
    test = datos.loc[datos[COLUMNA_TIEMPO] >= corte]
    resultados = validar_datasets(train, test)

    salida = REPORTS_DIR / "data-validation"
    salida.mkdir(parents=True, exist_ok=True)
    bloqueantes = []
    for nombre, resultado in resultados.items():
        reporte = salida / f"{nombre}.html"
        reporte.unlink(missing_ok=True)
        resultado.save_as_html(str(reporte))
        fallidos = checks_fallidos(resultado)
        severidad = "BLOQUEANTE" if nombre in SUITES_BLOQUEANTES else "advertencia"
        print(
            f"[{severidad}] {nombre}: {len(fallidos)} check(s) sin pasar: "
            f"{', '.join(fallidos) if fallidos else 'ninguno'}. Reporte: {reporte}"
        )
        if nombre in SUITES_BLOQUEANTES:
            bloqueantes.extend(fallidos)

    if bloqueantes:
        print(f"Validación fallida: {len(bloqueantes)} check(s) bloqueante(s) sin pasar.")
        return 1
    print("Validación DeepChecks completada sin problemas bloqueantes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
