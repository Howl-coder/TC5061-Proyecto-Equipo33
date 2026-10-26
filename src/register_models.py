"""Register selected MLflow runs and promote them with the ``champion`` alias.

Example:
    python -m src.register_models --p1-run-id <RUN_ID> --p1-justification "..."
        --p2-run-id <RUN_ID> --p2-justification "..."
        --p3-run-id <RUN_ID> --p3-justification "..."
"""

import argparse
from datetime import UTC, datetime

from config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_PRODUCTION_ALIAS,
    MLFLOW_REGISTERED_MODELS,
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_REPORT,
)
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import mlflow.pyfunc

PREGUNTAS = {
    "p1": {
        "tag": "P1",
        "nombre": "Pronóstico de demanda por estación",
        "metricas": ("test_rmse", "test_mae", "test_r2"),
        "principal": "test_rmse",
        "menor_es_mejor": True,
    },
    "p2": {
        "tag": "P2",
        "nombre": "Riesgo de desbalance de estaciones",
        "metricas": ("test_recall", "test_pr_auc", "test_f1", "test_roc_auc"),
        "principal": "test_recall",
        "menor_es_mejor": False,
    },
    "p3": {
        "tag": "P3",
        "nombre": "Segmentación de cicloestaciones",
        "metricas": ("silhouette", "davies_bouldin", "calinski_harabasz"),
        "principal": "silhouette",
        "menor_es_mejor": False,
    },
}


def _argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for pregunta in PREGUNTAS:
        parser.add_argument(f"--{pregunta}-run-id", required=True)
        parser.add_argument(
            f"--{pregunta}-justification",
            required=True,
            help="Razonamiento de selección tras comparar los runs en MLflow.",
        )
    return parser.parse_args()


def _validate_selected_run(client: MlflowClient, run_id: str, key: str, experiment_id: str):
    run = client.get_run(run_id)
    metadata = PREGUNTAS[key]

    if run.info.status != "FINISHED":
        raise ValueError(f"El run {run_id} no terminó correctamente: {run.info.status}.")
    if run.info.experiment_id != experiment_id:
        raise ValueError(f"El run {run_id} no pertenece a {MLFLOW_EXPERIMENT_NAME}.")
    if run.data.tags.get("question") != metadata["tag"]:
        raise ValueError(
            f"El run {run_id} tiene question={run.data.tags.get('question')!r}; "
            f"se esperaba {metadata['tag']}."
        )
    if run.data.tags.get("stage_of_work") not in {"baseline", "candidate"}:
        raise ValueError(f"El run {run_id} no es baseline ni candidate.")

    commit = run.data.tags.get("git_commit")
    dirty = run.data.tags.get("git_dirty")
    data_version = run.data.params.get("data_version")
    if not commit or commit == "desconocido":
        raise ValueError(f"El run {run_id} no tiene un commit Git identificable.")
    if dirty != "False":
        raise ValueError(f"El run {run_id} no se generó con un árbol de Git limpio.")
    if not data_version:
        raise ValueError(f"El run {run_id} no tiene data_version.")

    missing_metrics = [name for name in metadata["metricas"] if name not in run.data.metrics]
    if missing_metrics:
        raise ValueError(f"El run {run_id} no tiene las métricas requeridas: {missing_metrics}.")

    model_uri = f"runs:/{run_id}/modelo"
    model = mlflow.pyfunc.load_model(model_uri)
    if model.metadata.signature is None:
        raise ValueError(f"El modelo del run {run_id} no tiene firma MLflow.")

    return run


def _ensure_registered_model(client: MlflowClient, name: str, description: str) -> None:
    try:
        client.get_registered_model(name)
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        client.create_registered_model(name, description=description)


def _version_for_run(client: MlflowClient, name: str, run) -> str:
    existing = client.search_model_versions(f"name='{name}'")
    for version in existing:
        if version.run_id == run.info.run_id:
            return version.version

    registered = mlflow.register_model(
        f"runs:/{run.info.run_id}/modelo",
        name,
        await_registration_for=300,
    )
    return registered.version


def _training_seconds(run) -> float:
    if run.info.start_time is None or run.info.end_time is None:
        return 0.0
    return max(0, run.info.end_time - run.info.start_time) / 1000


def _format_metric(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def _comparison_runs(client: MlflowClient, experiment_id: str, question: str):
    return client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.question = '{question}'",
        max_results=1000,
    )


def _render_report(client: MlflowClient, experiment_id: str, selected: dict) -> str:
    lines = [
        "# Parte C — Model Registry y reporte",
        "",
        f"Promoción registrada: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        (
            f"Se usa el alias MLflow `{MLFLOW_PRODUCTION_ALIAS}` en vez de las etapas "
            "legacy (`Production`), porque los aliases son el mecanismo vigente y permiten "
            "mover el nombre de producción sin crear otra versión del modelo."
        ),
        "",
        (
            "La promoción registra cada modelo desde su run de origen; los tags de la versión "
            "conservan experimento, run, commit, etapa y versión de datos. La duración de "
            "entrenamiento se reporta como referencia de costo; el costo de inferencia no se "
            "midió en esta entrega."
        ),
        "",
    ]

    for key, metadata in PREGUNTAS.items():
        item = selected[key]
        run = item["run"]
        lines.extend(
            [
                f"## {metadata['tag']} — {metadata['nombre']}",
                "",
                f"- **Modelo registrado:** `{item['model_name']}` versión `{item['version']}`.",
                f"- **Alias de producción:** `{item['model_name']}@{MLFLOW_PRODUCTION_ALIAS}`.",
                f"- **Run de origen:** `{run.info.run_id}` (`{run.data.tags.get('mlflow.runName', '')}`).",
                (
                    f"- **Familia / etapa:** `{run.data.tags.get('model_family', 'desconocida')}` / "
                    f"`{run.data.tags.get('stage_of_work', 'desconocida')}`."
                ),
                (
                    f"- **Linaje:** experimento `{MLFLOW_EXPERIMENT_NAME}`, commit "
                    f"`{run.data.tags.get('git_commit')}`, datos "
                    f"`{run.data.params.get('data_version')}`."
                ),
                f"- **Duración de entrenamiento:** {_training_seconds(run):.1f} s.",
                f"- **Justificación de selección:** {item['justification']}",
                "",
                "| Run | Etapa | Familia | "
                + " | ".join(metadata["metricas"])
                + " | Entrenamiento (s) |",
                "|---|---|---|" + "---|" * len(metadata["metricas"]) + "---:|",
            ]
        )
        comparison = _comparison_runs(client, experiment_id, metadata["tag"])
        comparison.sort(
            key=lambda row: row.data.metrics.get(metadata["principal"], float("-inf")),
            reverse=not metadata["menor_es_mejor"],
        )
        for row in comparison:
            if row.data.tags.get("stage_of_work") not in {"baseline", "candidate"}:
                continue
            run_name = row.data.tags.get("mlflow.runName", row.info.run_id)
            metric_cells = " | ".join(
                _format_metric(row.data.metrics.get(metric)) for metric in metadata["metricas"]
            )
            selected_marker = " **(seleccionado)**" if row.info.run_id == run.info.run_id else ""
            lines.append(
                f"| `{run_name}`{selected_marker} | {row.data.tags.get('stage_of_work')} "
                f"| {row.data.tags.get('model_family', 'desconocida')} | {metric_cells} "
                f"| {_training_seconds(row):.1f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Verificación",
            "",
            (
                "Los tres modelos se cargaron por su URI de versión y, después de asignar el alias, "
                "se volvieron a cargar mediante `models:/<nombre>@champion`. No se registran datos "
                "personales de usuarios de MiBici."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _argparse()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(
            f"No existe el experimento {MLFLOW_EXPERIMENT_NAME!r} en {MLFLOW_TRACKING_URI}."
        )

    selected = {}
    used_run_ids = set()
    for key, metadata in PREGUNTAS.items():
        run_id = getattr(args, f"{key}_run_id")
        justification = getattr(args, f"{key}_justification").strip()
        if not justification:
            raise ValueError(f"Agrega una justificación para {metadata['tag']}.")
        if run_id in used_run_ids:
            raise ValueError("Cada pregunta debe usar un run de origen distinto.")
        used_run_ids.add(run_id)

        run = _validate_selected_run(client, run_id, key, experiment.experiment_id)
        selected[key] = {
            "run": run,
            "model_name": MLFLOW_REGISTERED_MODELS[key],
            "justification": justification,
        }

    # Complete validation and loadability checks before changing any production alias.
    for key, metadata in PREGUNTAS.items():
        item = selected[key]
        name = item["model_name"]
        run = item["run"]
        model_description = (
            f"{metadata['nombre']} para MiBici Equipo 33. La versión champion se selecciona "
            "con evaluación temporal de septiembre de 2026; consultar el reporte del proyecto."
        )
        _ensure_registered_model(client, name, model_description)
        client.update_registered_model(name, description=model_description)
        version = _version_for_run(client, name, run)
        item["version"] = version
        version_description = (
            f"Origen: run {run.info.run_id}; commit {run.data.tags['git_commit']}; "
            f"data_version {run.data.params['data_version']}. "
            f"Justificación: {item['justification']}"
        )
        client.update_model_version(
            name=name,
            version=version,
            description=version_description,
        )
        version_tags = {
            "question": metadata["tag"],
            "model_family": run.data.tags.get("model_family", "desconocida"),
            "stage_of_work": run.data.tags["stage_of_work"],
            "source_experiment": MLFLOW_EXPERIMENT_NAME,
            "source_experiment_id": run.info.experiment_id,
            "source_run_id": run.info.run_id,
            "git_commit": run.data.tags["git_commit"],
            "git_dirty": run.data.tags["git_dirty"],
            "data_version": run.data.params["data_version"],
        }
        for tag, value in version_tags.items():
            client.set_model_version_tag(name, version, tag, str(value))

        version_uri = f"models:/{name}/{version}"
        version_model = mlflow.pyfunc.load_model(version_uri)
        if version_model.metadata.signature is None:
            raise ValueError(f"La versión registrada {version_uri} no conserva su firma.")

    for item in selected.values():
        client.set_registered_model_alias(
            item["model_name"],
            MLFLOW_PRODUCTION_ALIAS,
            item["version"],
        )

    for item in selected.values():
        alias_uri = f"models:/{item['model_name']}@{MLFLOW_PRODUCTION_ALIAS}"
        alias_model = mlflow.pyfunc.load_model(alias_uri)
        if alias_model.metadata.signature is None:
            raise ValueError(f"No se pudo verificar la firma del modelo {alias_uri}.")
        print(f"Verificado: {alias_uri} (run {item['run'].info.run_id})")

    report = _render_report(client, experiment.experiment_id, selected)
    MODEL_REGISTRY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY_REPORT.write_text(report, encoding="utf-8")
    print(f"Reporte generado: {MODEL_REGISTRY_REPORT}")


if __name__ == "__main__":
    main()
