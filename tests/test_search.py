import pandas as pd
import pytest

from src.search import _partir_validacion


@pytest.fixture
def historia_horaria():
    return pd.DataFrame(
        {"timestamp_hora": pd.date_range("2026-06-25", "2026-08-05", freq="h")}
    )


def test_partir_validacion_separa_ventana_de_julio_sin_traslape(historia_horaria):
    fit, validacion = _partir_validacion(historia_horaria)

    assert fit["timestamp_hora"].max() == pd.Timestamp("2026-06-30 23:00")
    assert validacion["timestamp_hora"].min() == pd.Timestamp("2026-07-01 00:00")
    assert validacion["timestamp_hora"].max() == pd.Timestamp("2026-07-31 23:00")
    assert len(validacion) == 31 * 24


def test_partir_validacion_aplica_purga_antes_de_validacion(historia_horaria):
    fit, validacion = _partir_validacion(historia_horaria, purge_hours=3)

    assert fit["timestamp_hora"].max() == pd.Timestamp("2026-06-30 20:00")
    assert validacion["timestamp_hora"].min() == pd.Timestamp("2026-07-01 00:00")


def test_partir_validacion_rechaza_purga_negativa(historia_horaria):
    with pytest.raises(ValueError, match="negativo"):
        _partir_validacion(historia_horaria, purge_hours=-1)


def test_partir_validacion_falla_sin_historia_previa():
    solo_julio = pd.DataFrame(
        {"timestamp_hora": pd.date_range("2026-07-01", "2026-07-10", freq="h")}
    )

    with pytest.raises(ValueError, match="suficientes filas"):
        _partir_validacion(solo_julio)
