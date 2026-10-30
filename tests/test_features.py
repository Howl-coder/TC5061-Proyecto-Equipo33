import pandas as pd

from src.features import (
    calcular_features_temporales_y_lags,
    generar_malla_demanda_horaria,
)


def test_generar_malla_rellena_sin_viajes_con_cero():
    viajes = pd.DataFrame(
        {
            "Inicio_del_viaje": pd.to_datetime(["2026-04-01 00:10"]),
            "Fin_del_viaje": pd.to_datetime(["2026-04-01 01:15"]),
            "Origen_Id": [1],
            "Destino_Id": [2],
        }
    )

    malla = generar_malla_demanda_horaria(viajes, [1, 2])

    assert len(malla) == 4
    assert malla["demanda_salidas"].sum() == 1
    assert malla["flujo_llegadas"].sum() == 1
    assert malla.loc[malla["Origen_Id"] == 2, "demanda_salidas"].eq(0).all()


def test_calcular_features_respeta_rezagos_y_minimo_de_historia():
    malla = pd.DataFrame(
        {
            "timestamp_hora": pd.date_range("2026-04-01", periods=170, freq="h"),
            "Origen_Id": 1,
            "demanda_salidas": range(170),
            "flujo_llegadas": range(170),
        }
    )

    features = calcular_features_temporales_y_lags(malla)

    primera = features.iloc[0]
    assert primera["timestamp_hora"] == pd.Timestamp("2026-04-08 00:00")
    assert primera["lag_salidas_168h"] == 0
    assert primera["lag_salidas_1h"] == 167
    assert primera["rolling_mean_6h"] == 164.5


def test_calcular_features_devuelve_vacio_si_no_hay_historia_suficiente():
    malla = pd.DataFrame(
        {
            "timestamp_hora": pd.date_range("2026-04-01", periods=168, freq="h"),
            "Origen_Id": 1,
            "demanda_salidas": range(168),
            "flujo_llegadas": range(168),
        }
    )

    assert calcular_features_temporales_y_lags(malla).empty
