import pandas as pd
import pytest

from src.features import (
    agregar_features_y_target_desbalance,
    calcular_features_temporales_y_lags,
    construir_perfil_estaciones,
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


def test_generar_malla_no_modifica_conteos_con_varias_estaciones_y_viajes_largos():
    viajes = pd.DataFrame(
        {
            "Inicio_del_viaje": pd.to_datetime(["2026-04-01 00:05", "2026-04-01 00:50"]),
            "Fin_del_viaje": pd.to_datetime(["2026-04-01 02:10", "2026-04-01 00:55"]),
            "Origen_Id": [1, 1],
            "Destino_Id": [3, 1],
        }
    )

    malla = generar_malla_demanda_horaria(viajes, [1, 2, 3])

    # 3 horas (00h..02h, la llegada más tardía extiende el rango) x 3 estaciones
    assert len(malla) == 9
    fila = malla.set_index(["Origen_Id", "timestamp_hora"])
    assert fila.loc[(1, pd.Timestamp("2026-04-01 00:00")), "demanda_salidas"] == 2
    assert fila.loc[(1, pd.Timestamp("2026-04-01 00:00")), "flujo_llegadas"] == 1
    assert fila.loc[(3, pd.Timestamp("2026-04-01 02:00")), "flujo_llegadas"] == 1
    assert malla.loc[malla["Origen_Id"] == 2, ["demanda_salidas", "flujo_llegadas"]].eq(0).all().all()


def _features_balance(balances_por_estacion: dict) -> pd.DataFrame:
    filas = [
        {
            "timestamp_hora": pd.Timestamp("2026-04-08") + pd.Timedelta(hours=h),
            "Origen_Id": estacion,
            "balance_neto_h": balance,
        }
        for estacion, balances in balances_por_estacion.items()
        for h, balance in enumerate(balances)
    ]
    return pd.DataFrame(filas)


def test_target_desbalance_usa_balance_futuro_y_rezagos_por_estacion():
    features = _features_balance({1: [0, 1, 2, -3, 0, 0]})
    estaciones = pd.DataFrame({"id": [1], "dpcapacity": [10]})

    resultado = agregar_features_y_target_desbalance(
        features, estaciones, horizonte_h=2, umbral_fraccion_capacidad=0.3
    )

    # Se pierden 2 filas al inicio (rezagos) y 2 al final (horizonte sin futuro completo)
    assert resultado["timestamp_hora"].tolist() == [
        pd.Timestamp("2026-04-08 02:00"),
        pd.Timestamp("2026-04-08 03:00"),
    ]
    primera = resultado.iloc[0]
    assert primera["lag_balance_1h"] == 1
    assert primera["lag_balance_2h"] == 0
    assert primera["balance_acum_3h"] == 3
    # |(-3) + 0| = 3 alcanza exactamente el umbral 0.3 * 10
    assert resultado["en_riesgo_desbalance"].tolist() == [1, 0]


def test_target_desbalance_descarta_estaciones_fuera_del_catalogo():
    features = _features_balance({1: [0] * 6, 99: [5] * 6})
    estaciones = pd.DataFrame({"id": [1], "dpcapacity": [10]})

    resultado = agregar_features_y_target_desbalance(
        features, estaciones, horizonte_h=1, umbral_fraccion_capacidad=0.3
    )

    assert set(resultado["Origen_Id"]) == {1}
    assert resultado["en_riesgo_desbalance"].eq(0).all()


def test_target_desbalance_devuelve_vacio_si_la_serie_es_mas_corta_que_el_horizonte():
    features = _features_balance({1: [4, -4, 4]})
    estaciones = pd.DataFrame({"id": [1], "dpcapacity": [10]})

    resultado = agregar_features_y_target_desbalance(
        features, estaciones, horizonte_h=3, umbral_fraccion_capacidad=0.3
    )

    assert resultado.empty


@pytest.fixture
def viajes_y_estaciones():
    viajes = pd.DataFrame(
        {
            "Inicio_del_viaje": pd.to_datetime(
                [
                    "2026-04-01 08:15",  # miércoles
                    "2026-04-01 08:45",  # viaje circular
                    "2026-04-04 18:00",  # sábado
                    "2026-04-02 09:00",
                    "2026-04-02 10:00",
                    "2026-05-01 08:00",  # posterior a fecha_fin
                ]
            ),
            "Origen_Id": [1, 1, 1, 2, 2, 1],
            "Destino_Id": [2, 1, 2, 1, 3, 2],
        }
    )
    estaciones = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "latitude": [20.6, 20.7, 20.65],
            "longitude": [-103.3, -103.4, -103.35],
            "dpcapacity": [10, 0, 5],
        }
    )
    return viajes, estaciones


def test_perfil_estaciones_calcula_proporciones_y_uso(viajes_y_estaciones):
    viajes, estaciones = viajes_y_estaciones

    perfil = construir_perfil_estaciones(viajes, estaciones, "2026-04-30 23:59:59").set_index("id")

    columnas_hora = [f"prop_hora_{h:02d}" for h in range(24)]
    assert perfil[columnas_hora].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])

    estacion_1 = perfil.loc[1]
    assert estacion_1["salidas"] == 3  # el viaje de mayo queda fuera
    assert estacion_1["llegadas"] == 2
    assert estacion_1["prop_hora_08"] == pytest.approx(2 / 3)
    assert estacion_1["prop_fin_de_semana"] == pytest.approx(1 / 3)
    assert estacion_1["prop_circulares"] == pytest.approx(1 / 3)
    assert estacion_1["prop_salidas"] == pytest.approx(3 / 5)
    assert estacion_1["viajes_por_dia"] == pytest.approx(1.0)  # 3 salidas / 3 días
    assert estacion_1["uso_por_capacidad"] == pytest.approx(0.1)


def test_perfil_estaciones_casos_limite(viajes_y_estaciones):
    viajes, estaciones = viajes_y_estaciones

    perfil = construir_perfil_estaciones(viajes, estaciones, "2026-04-30 23:59:59").set_index("id")

    # La estación 3 solo recibe llegadas: no tiene perfil de salidas
    assert sorted(perfil.index) == [1, 2]
    # Capacidad 0 se acota a 1 para evitar división entre cero
    assert perfil.loc[2, "uso_por_capacidad"] == pytest.approx(2 / 3)
    assert perfil.notna().all().all()
