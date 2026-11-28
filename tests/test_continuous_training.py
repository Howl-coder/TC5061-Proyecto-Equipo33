import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import continuous_training
from src.continuous_training import (
    debe_reentrenar,
    descubrir_archivos_publicos,
    evaluar_gates_promocion,
)
from src.p1_schema import FEATURES, TARGET


def _pagina(periodos):
    enlaces = [
        f'<a href="/site/assets/files/{index}/datos_abiertos_{periodo.replace("-", "_")}.csv">'
        "CSV</a>"
        for index, periodo in enumerate(periodos, start=1)
    ]
    enlaces.append('<a href="/site/assets/files/99/nomenclatura_2026_10.csv">Catálogo</a>')
    return "".join(enlaces)


def test_descubre_ventana_mensual_reciente_y_catalogo():
    periodos = [
        "2026-05",
        "2026-06",
        "2026-07",
        "2026-08",
        "2026-09",
        "2026-10",
        "2026-11",
    ]

    archivos, catalogo = descubrir_archivos_publicos(_pagina(periodos))

    assert [archivo["periodo"] for archivo in archivos] == periodos[-6:]
    assert archivos[-1]["nombre"] == "datos_abiertos_2026_11.csv"
    assert archivos[-1]["url"].startswith("https://www.mibici.net/")
    assert catalogo["nombre"] == "nomenclatura_2026_10.csv"


def test_rechaza_meses_no_consecutivos():
    periodos = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-09", "2026-10"]

    with pytest.raises(ValueError, match="no son consecutivos"):
        descubrir_archivos_publicos(_pagina(periodos))


def test_rechaza_enlaces_de_datos_fuera_del_dominio_oficial():
    html = (
        '<a href="https://evil.example/site/assets/files/1/'
        'datos_abiertos_2026_10.csv">CSV</a>'
        '<a href="/site/assets/files/99/nomenclatura_2026_10.csv">Catálogo</a>'
    )
    html += _pagina(["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"])

    with pytest.raises(ValueError, match="dominio HTTPS permitido"):
        descubrir_archivos_publicos(html)


@pytest.mark.parametrize(
    ("drift_share", "esperado"),
    [(0.39, False), (0.4, True), (1.0, True)],
)
def test_umbral_de_drift_dispara_reentrenamiento(drift_share, esperado):
    assert debe_reentrenar(drift_share) is esperado


def test_umbral_de_drift_rechaza_valores_invalidos():
    with pytest.raises(ValueError, match="entre 0 y 1"):
        debe_reentrenar(1.01)


def test_evidently_guarda_reporte_y_calcula_proporcion_de_drift(tmp_path):
    cantidad = 120
    referencia = pd.DataFrame(
        {
            columna: (
                np.resize([0, 1], cantidad)
                if columna == "es_fin_de_semana"
                else np.linspace(0, 1, cantidad) + indice
            )
            for indice, columna in enumerate([*FEATURES, TARGET])
        }
    )
    actual = referencia.copy()
    for columna in [*FEATURES, TARGET]:
        if columna != "es_fin_de_semana":
            actual[columna] += 2
    ruta_html = tmp_path / "drift.html"

    resultado = continuous_training._evaluar_drift(referencia, actual, ruta_html)

    assert 0 <= resultado["drift_share"] <= 1
    assert resultado["drifted_columns"] >= 0
    assert ruta_html.is_file()
    assert json.dumps(resultado["json"], ensure_ascii=False)


def test_gates_solo_aprueban_mejora_estadistica_segmentada_y_con_costo_acotado():
    gates = evaluar_gates_promocion(
        rmse_candidata=1.4,
        rmse_champion=1.5,
        limite_superior_ic_delta=-0.01,
        max_delta_rmse_estacion=0.2,
        segundos_entrenamiento=600,
    )

    assert all(gates.values())


@pytest.mark.parametrize(
    ("rmse", "ic_superior", "delta_estacion", "segundos", "gate"),
    [
        (2.01, -0.01, 0.1, 600, "absolute_quality"),
        (1.49, -0.01, 0.1, 600, "relative_quality"),
        (1.4, 0.0, 0.1, 600, "statistical_quality"),
        (1.4, -0.01, 0.26, 600, "segment_quality"),
        (1.4, -0.01, 0.1, 901, "training_cost"),
    ],
)
def test_gates_rechazan_candidatas_que_exceden_un_limite(
    rmse, ic_superior, delta_estacion, segundos, gate
):
    gates = evaluar_gates_promocion(
        rmse_candidata=rmse,
        rmse_champion=1.5,
        limite_superior_ic_delta=ic_superior,
        max_delta_rmse_estacion=delta_estacion,
        segundos_entrenamiento=segundos,
    )

    assert not gates[gate]


def test_hash_solo_se_considera_procesado_si_el_run_termino():
    class Cliente:
        def __init__(self, status):
            self.status = status

        def search_runs(self, experiment_ids, filter_string, max_results):
            assert filter_string == "tags.ct_fingerprint = 'fingerprint'"
            assert max_results == 1000
            return [SimpleNamespace(info=SimpleNamespace(status=self.status))]

    assert not continuous_training._run_id_cubierto(Cliente("FAILED"), "experiment", "fingerprint")
    assert continuous_training._run_id_cubierto(Cliente("FINISHED"), "experiment", "fingerprint")


def test_rollback_no_cambia_alias_si_no_coincide_version_esperada(tmp_path, monkeypatch):
    class Cliente:
        def __init__(self, tracking_uri):
            self.alias_calls = 0

        def get_model_version_by_alias(self, name, alias):
            return SimpleNamespace(version="7")

        def set_registered_model_alias(self, *args):
            self.alias_calls += 1

    cliente = Cliente("https://mlflow.example")
    monkeypatch.setattr(continuous_training, "MlflowClient", lambda **kwargs: cliente)

    with pytest.raises(RuntimeError, match="apunta a 7"):
        continuous_training.revertir_alias("5", "6", tmp_path)

    assert cliente.alias_calls == 0
