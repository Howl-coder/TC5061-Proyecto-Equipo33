"""Búsqueda de hiperparámetros con Optuna + MLflow nested runs para P1, P2 y P3.

Uso:
    python -m src.search --pregunta p1
    python -m src.search --pregunta p2 --n-trials 8
    python -m src.search --pregunta p3
"""
import argparse
import contextlib

import mlflow
import optuna

from config import RANDOM_STATE
from src.tracking import configurar_experimento, params_reproducibilidad, tags_run

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS_DEFAULT = 5


def _run_o_activo(nombre: str):
    """Usa el run activo de `mlflow run` si ya existe; si no, crea uno nuevo."""
    if mlflow.active_run():
        mlflow.set_tag("mlflow.runName", nombre)
        return contextlib.nullcontext()
    return mlflow.start_run(run_name=nombre)


# ── P1 ──────────────────────────────────────────────────────────────────────

def _buscar_p1(params_base: dict, n_trials: int) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    from src.train_regresion import cargar_datos, ejecutar_run

    train, test = cargar_datos()

    def objective(trial) -> float:
        hiperparams = {
            "max_iter": trial.suggest_int("max_iter", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        }
        metricas = ejecutar_run(
            f"P1-candidate-histgb-t{trial.number}",
            "HistGradientBoostingRegressor",
            train, test, params_base,
            modelo=HistGradientBoostingRegressor(**hiperparams, random_state=RANDOM_STATE),
            stage="candidate",
            nested=True,
        )
        return metricas["test_rmse"]

    with _run_o_activo("P1-search-histgb"):
        mlflow.set_tags(tags_run("P1", "HistGradientBoostingRegressor", stage="search"))
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="minimize", study_name="P1-histgb")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_test_rmse", study.best_value)
        print(f"\nP1 mejor trial #{study.best_trial.number}: RMSE={study.best_value:.4f}")
        print(f"  Params: {study.best_params}")


# ── P2 ──────────────────────────────────────────────────────────────────────

def _buscar_p2(params_base: dict, n_trials: int) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.train_clasificacion import MODELOS, cargar_datos, ejecutar_run

    train, test = cargar_datos()

    def objective(trial) -> float:
        tipo = trial.suggest_categorical("tipo_modelo", ["logreg", "histgb"])

        if tipo == "logreg":
            hiperparams = {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "class_weight": "balanced",
                "max_iter": trial.suggest_int("max_iter", 500, 2000),
            }
            tipo_modelo, construir = MODELOS["logreg"]
        else:
            hiperparams = {
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "class_weight": "balanced",
            }
            tipo_modelo, construir = MODELOS["histgb"]

        metricas = ejecutar_run(
            f"P2-candidate-{tipo}-t{trial.number}",
            tipo_modelo,
            construir(hiperparams),
            hiperparams,
            train, test, params_base,
            stage="candidate",
            nested=True,
        )
        return -metricas["test_f1"]

    with _run_o_activo("P2-search-mixed"):
        mlflow.set_tags(tags_run("P2", "LogisticRegression+HistGBClassifier", stage="search"))
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="minimize", study_name="P2-mixed")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_test_f1", -study.best_value)
        print(f"\nP2 mejor trial #{study.best_trial.number}: F1={-study.best_value:.4f}")
        print(f"  Params: {study.best_params}")


# ── P3 ──────────────────────────────────────────────────────────────────────

def _buscar_p3(params_base: dict, n_trials: int) -> None:
    from src.train_clustering import cargar_datos, entrenar

    perfil = cargar_datos()

    def objective(trial) -> float:
        hiperparams = {
            "k_min": trial.suggest_int("k_min", 2, 4),
            "k_max": trial.suggest_int("k_max", 8, 15),
            "n_init": trial.suggest_int("n_init", 5, 20),
        }
        metricas = entrenar(
            f"kmeans-t{trial.number}",
            hiperparams,
            perfil,
            params_base,
            stage="candidate",
            nested=True,
        )
        return -metricas["silhouette"]

    with _run_o_activo("P3-search-kmeans"):
        mlflow.set_tags(tags_run("P3", "KMeans", stage="search"))
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="minimize", study_name="P3-kmeans")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_silhouette", -study.best_value)
        print(f"\nP3 mejor trial #{study.best_trial.number}: Silhouette={-study.best_value:.4f}")
        print(f"  Params: {study.best_params}")


# ── Dispatcher ───────────────────────────────────────────────────────────────

BUSCADORES = {"p1": _buscar_p1, "p2": _buscar_p2, "p3": _buscar_p3}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pregunta", required=True, choices=sorted(BUSCADORES))
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                        help=f"Número de trials de Optuna (default: {N_TRIALS_DEFAULT}).")
    args = parser.parse_args()

    configurar_experimento()
    params_base = params_reproducibilidad()
    BUSCADORES[args.pregunta](params_base, args.n_trials)


if __name__ == "__main__":
    main()
