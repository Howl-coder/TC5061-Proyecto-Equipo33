import numpy as np
import pandas as pd
import pytest

from src.ab_testing import bootstrap_delta_rmse_por_dia, metricas_regresion


def test_metricas_regresion_calcula_rmse_mae_r2_y_sesgo():
    metricas = metricas_regresion([1, 2, 3], [2, 2, 4])

    assert metricas["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metricas["mae"] == pytest.approx(2 / 3)
    assert metricas["r2"] == pytest.approx(0.0)
    assert metricas["sesgo_pred_menos_real"] == pytest.approx(2 / 3)


def test_bootstrap_pareado_por_dia_detecta_candidato_uniformemente_peor():
    fechas = pd.date_range("2026-09-01", periods=3, freq="D").repeat(2)
    y_real = np.zeros(6)
    pred_a = np.zeros(6)
    pred_b = np.ones(6)

    resultado = bootstrap_delta_rmse_por_dia(
        y_real, pred_a, pred_b, fechas, n_replicas=200, random_state=7
    )

    assert resultado["delta_observado_rmse_b_menos_a"] == pytest.approx(1.0)
    assert resultado["ic_95_inferior"] == pytest.approx(1.0)
    assert resultado["ic_95_superior"] == pytest.approx(1.0)
    assert resultado["fraccion_replicas_b_peor"] == 1.0
    assert resultado["bloques_dia"] == 3


def test_metricas_regresion_rechaza_vectores_vacios_y_con_nan():
    with pytest.raises(ValueError, match="igual tamaño"):
        metricas_regresion([], [])

    with pytest.raises(ValueError, match="finitos"):
        metricas_regresion([1, float("nan")], [1, 2])

    with pytest.raises(ValueError, match="finitos"):
        metricas_regresion([1, 2], [1, float("inf")])


def test_bootstrap_rechaza_un_solo_dia_y_longitudes_distintas():
    with pytest.raises(ValueError, match="igual tamaño"):
        bootstrap_delta_rmse_por_dia(
            [1, 2], [1], [1, 2], pd.date_range("2026-09-01", periods=2, freq="h")
        )

    with pytest.raises(ValueError, match="al menos dos bloques diarios"):
        bootstrap_delta_rmse_por_dia(
            [1, 2], [1, 2], [1, 2], pd.date_range("2026-09-01", periods=2, freq="h")
        )
