import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import mlflow
import pandas as pd

from src.dataset import generar_manifiesto

ROOT = Path(__file__).resolve().parents[1]


def test_mlproject_prepara_datos_y_registra_entrenamiento(tmp_path):
    shutil.copytree(ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for nombre in ("MLproject", "python_env.yaml", "config.py"):
        shutil.copy2(ROOT / nombre, tmp_path / nombre)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    archivos = []
    for mes in range(4, 10):
        archivo = raw / f"datos_abiertos_2026_{mes:02d}.csv"
        inicio = pd.Timestamp(f"2026-{mes:02d}-10 08:00:00")
        pd.DataFrame(
            {
                "Viaje_Id": [mes],
                "Inicio_del_viaje": [inicio],
                "Fin_del_viaje": [inicio + pd.Timedelta(minutes=15)],
                "Origen_Id": [1],
                "Destino_Id": [2],
            }
        ).to_csv(archivo, index=False)
        archivos.append(archivo)
    catalogo = raw / "nomenclatura_2026_09.csv"
    pd.DataFrame(
        {"id": [1, 2], "name": ["Centro", "Destino"], "latitude": [20.67, 20.68],
         "longitude": [-103.35, -103.36], "dpcapacity": [15, 15]}
    ).to_csv(catalogo, index=False)
    archivos.append(catalogo)
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "manifiesto_datos.json").write_text(
        json.dumps(generar_manifiesto(archivos)), encoding="utf-8"
    )

    tracking_uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    env = os.environ.copy()
    env.pop("MLFLOW_RUN_ID", None)
    env.pop("MLFLOW_EXPERIMENT_ID", None)
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    def ejecutar(entry_point, *parametros):
        comando = [
            sys.executable, "-m", "mlflow", "run", str(tmp_path),
            "-e", entry_point, "--experiment-name", "mibici-equipo33",
            "--env-manager", "local", *parametros,
        ]
        return subprocess.run(
            comando, cwd=tmp_path, env=env, text=True, capture_output=True, check=False
        )

    for entry_point, parametros in (
        ("ingest", ()),
        ("features", ()),
        ("train", ("-P", "pregunta=p1", "-P", "modelo=naive")),
    ):
        resultado = ejecutar(entry_point, *parametros)
        assert resultado.returncode == 0, resultado.stdout + resultado.stderr

    features = pd.read_parquet(tmp_path / "data" / "processed" / "features_demanda_p1.parquet")
    destino = features[features["Origen_Id"] == 2]
    assert not destino.empty
    assert destino["demanda_salidas"].eq(0).all()
    assert destino["flujo_llegadas"].sum() > 0

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experimento = client.get_experiment_by_name("mibici-equipo33")
    runs = client.search_runs([experimento.experiment_id])
    preparacion = {run.data.tags.get("etapa_pipeline"): run for run in runs}
    for etapa in ("ingest", "features"):
        run = preparacion[etapa]
        assert run.info.status == "FINISHED"
        assert len(run.data.params["archivo_sha256"]) == 64
        assert client.list_artifacts(run.info.run_id, "procedencia")
    entrenamientos = [run for run in runs if run.data.tags.get("question") == "P1"]
    assert len(entrenamientos) == 1
    run = entrenamientos[0]
    assert run.info.status == "FINISHED"
    assert "test_rmse" in run.data.metrics
    assert len(run.data.params["interim_sha256"]) == 64
    assert len(run.data.params["features_sha256"]) == 64
    assert run.data.params["python_version"] == ".".join(map(str, sys.version_info[:3]))
    assert run.data.params["mlflow_version"] == mlflow.__version__

    resultado = ejecutar("search", "-P", "pregunta=p1", "-P", "n_trials=1")
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    runs = client.search_runs([experimento.experiment_id])
    padres = [run for run in runs if run.data.tags.get("stage_of_work") == "search"]
    assert len(padres) == 1
    assert "best_validation_rmse" in padres[0].data.metrics
    assert padres[0].data.params["features_sha256"] == run.data.params["features_sha256"]
    assert any(run.data.tags.get("mlflow.parentRunId") == padres[0].info.run_id for run in runs)

    (tmp_path / "data" / "processed" / "features_demanda_p1.parquet").write_bytes(b"cambiado")
    resultado = ejecutar("train", "-P", "pregunta=p1", "-P", "modelo=naive")
    assert resultado.returncode != 0
    assert "Procedencia desactualizada" in resultado.stderr
