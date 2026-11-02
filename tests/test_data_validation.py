import numpy as np
import pandas as pd
import pytest

pytest.importorskip("deepchecks")

from src.data_validation import (  # noqa: E402
    checks_fallidos,
    suite_integridad,
    suite_particion,
    validar_datasets,
    validar_esquema,
    _dataset,
)
from src.p1_schema import FEATURES, TARGET  # noqa: E402


def _features(inicio: str, horas: int, estaciones=(1, 2, 3), semilla=0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    tiempos = pd.date_range(inicio, periods=horas, freq="h")
    df = pd.MultiIndex.from_product(
        [estaciones, tiempos], names=["Origen_Id", "timestamp_hora"]
    ).to_frame(index=False)
    hora, dia = df["timestamp_hora"].dt.hour, df["timestamp_hora"].dt.dayofweek
    df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    df["dia_sin"] = np.sin(2 * np.pi * dia / 7)
    df["dia_cos"] = np.cos(2 * np.pi * dia / 7)
    df["es_fin_de_semana"] = (dia >= 5).astype(int)
    for col in FEATURES[5:]:
        df[col] = rng.poisson(3, len(df)).astype(float)
    df[TARGET] = rng.poisson(3, len(df))
    return df


@pytest.fixture
def train():
    return _features("2026-08-01", 14 * 24)


@pytest.fixture
def test():
    return _features("2026-09-01", 7 * 24, semilla=1)


def test_esquema_valido_no_reporta_problemas(train):
    assert validar_esquema(train) == []


def test_esquema_detecta_columnas_faltantes(train):
    problemas = validar_esquema(train.drop(columns=["lag_salidas_1h"]))

    assert problemas == ["Faltan columnas requeridas: ['lag_salidas_1h']"]


def test_esquema_detecta_tipos_target_negativo_y_llave_repetida(train):
    malo = train.copy()
    malo["timestamp_hora"] = malo["timestamp_hora"].astype(str)
    malo["hora_sin"] = malo["hora_sin"].astype(str)
    malo.loc[0, TARGET] = -1
    malo = pd.concat([malo, malo.iloc[[1]]])

    problemas = " | ".join(validar_esquema(malo))

    assert "'timestamp_hora' debe ser datetime64" in problemas
    assert "Features no numéricas: ['hora_sin']" in problemas
    assert "valores negativos" in problemas
    assert "1 fila(s) repiten la llave" in problemas


def test_validar_datasets_rechaza_esquema_invalido(train, test):
    with pytest.raises(ValueError, match="Esquema inválido en test"):
        validar_datasets(train, test.drop(columns=[TARGET]))


def test_validar_datasets_rechaza_conjuntos_vacios(train, test):
    with pytest.raises(ValueError, match="deben contener filas"):
        validar_datasets(train, test.iloc[:0])


def test_integridad_pasa_con_datos_limpios(train, test):
    resultado = suite_integridad().run(train_dataset=_dataset(train), test_dataset=_dataset(test))

    assert checks_fallidos(resultado) == []


def test_integridad_detecta_nulos_y_duplicados(train, test):
    sucio = pd.concat([train, train.iloc[:5]], ignore_index=True)
    sucio.loc[10:20, "lag_salidas_24h"] = np.nan

    resultado = suite_integridad().run(train_dataset=_dataset(sucio), test_dataset=_dataset(test))

    fallidos = checks_fallidos(resultado)
    assert "PercentOfNulls - Train Dataset" in fallidos
    assert "Data Duplicates - Train Dataset" in fallidos
    assert not any(nombre.endswith("Test Dataset") for nombre in fallidos)


def test_particion_detecta_fuga_temporal(train):
    traslapado = _features("2026-08-10", 7 * 24, semilla=2)

    resultado = suite_particion().run(
        train_dataset=_dataset(train), test_dataset=_dataset(traslapado)
    )

    assert "Date Train-Test Leakage (overlap)" in checks_fallidos(resultado)


def test_particion_pasa_con_corte_temporal_correcto(train, test):
    resultado = suite_particion().run(train_dataset=_dataset(train), test_dataset=_dataset(test))

    assert checks_fallidos(resultado) == []
