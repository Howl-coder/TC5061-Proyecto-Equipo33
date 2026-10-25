"""P1 (Regresión): baselines de demanda horaria de salidas por estación, registrados en MLflow.

Uso: python -m src.train_regresion
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    BASE_DIR,
    FECHA_INICIO_TEST,
    FIGURES_DIR,
    HIPERPARAMS_POR_DEFECTO,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RUTA_MANIFIESTO_DATOS,
)
from src.tracking import (
    SKOPS_TIPOS_CONFIABLES,
    configurar_experimento,
    params_reproducibilidad,
    tags_run,
)

TARGET = "demanda_salidas"
FEATURES = [
    "hora_sin", "hora_cos", "dia_sin", "dia_cos", "es_fin_de_semana",
    "lag_salidas_1h", "lag_salidas_2h", "lag_salidas_3h",
    "lag_salidas_24h", "lag_salidas_168h", "rolling_mean_6h",
]
TIPOS_NAIVE = {"lag_salidas_1h": "naive_persistencia", "lag_salidas_168h": "naive_estacional"}


def metricas_regresion(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def graficar_real_vs_pred(test: pd.DataFrame, y_pred, nombre: str):
    """Serie real vs. predicha de la primera semana de test en la estación de mayor demanda."""
    test = test.assign(pred=y_pred)
    estacion = test.groupby("Origen_Id")[TARGET].sum().idxmax()
    serie = test[test["Origen_Id"] == estacion].head(24 * 7)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(serie["timestamp_hora"], serie[TARGET], label="Real", lw=1.5)
    ax.plot(serie["timestamp_hora"], serie["pred"], label="Predicción", lw=1.2, ls="--")
    ax.set_title(f"P1 {nombre} — estación {estacion}, primera semana de test")
    ax.set_ylabel("Salidas por hora")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def ejecutar_run(
    nombre_run, tipo_modelo, train, test, params_base,
    modelo=None, columna_naive=None, stage="baseline", nested=False,
) -> dict:
    with mlflow.start_run(run_name=nombre_run, nested=nested):
        mlflow.set_tags(tags_run("P1", tipo_modelo, stage=stage))
        mlflow.log_params(params_base)
        mlflow.log_params({
            "target": TARGET,
            "horizonte_h": 1,
            "n_train": len(train),
            "n_test": len(test),
            "n_estaciones": test["Origen_Id"].nunique(),
        })

        if modelo is not None:
            mlflow.log_params({"features": ",".join(FEATURES), **modelo.get_params()})
            modelo.fit(train[FEATURES], train[TARGET])
            y_pred = np.clip(modelo.predict(test[FEATURES]), 0, None)
            mlflow.log_metrics({f"train_{k}": v for k, v in metricas_regresion(
                train[TARGET], modelo.predict(train[FEATURES])).items()})

            signature = infer_signature(train[FEATURES], modelo.predict(train[FEATURES]))
            mlflow.sklearn.log_model(
                modelo, name="modelo",
                signature=signature,
                input_example=train[FEATURES].iloc[:3],
                skops_trusted_types=SKOPS_TIPOS_CONFIABLES,
            )
        else:
            mlflow.log_param("columna_naive", columna_naive)
            y_pred = test[columna_naive].to_numpy()

        metricas = {f"test_{k}": v for k, v in metricas_regresion(test[TARGET], y_pred).items()}
        mlflow.log_metrics(metricas)

        dataset_train = mlflow.data.from_pandas(
            train[FEATURES + [TARGET]],
            source=str(PROCESSED_DATA_DIR / "features_demanda_p1.parquet"),
            name="mibici-train-p1",
            targets=TARGET,
        )
        mlflow.log_input(dataset_train, context="training")

        fig = graficar_real_vs_pred(test, y_pred, nombre_run)
        ruta_fig = FIGURES_DIR / f"{nombre_run}_real_vs_pred.png"
        fig.savefig(ruta_fig, dpi=120, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(ruta_fig), artifact_path="figuras")

        mlflow.log_artifact(str(RUTA_MANIFIESTO_DATOS), artifact_path="datos")
        mlflow.log_artifact(str(BASE_DIR / "config.py"), artifact_path="config")

        print(f"{nombre_run}: {metricas}")
    return metricas


def cargar_datos():
    """Devuelve ``(train, test)`` con la partición temporal de ``config.FECHA_INICIO_TEST``."""
    ruta = PROCESSED_DATA_DIR / "features_demanda_p1.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"Primero corre src.dataset y src.features para generar {ruta}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ruta)
    corte = pd.Timestamp(FECHA_INICIO_TEST)
    return df[df["timestamp_hora"] < corte], df[df["timestamp_hora"] >= corte]


def entrenar(modelo: str, hiperparams: dict, datos, params_base: dict, stage: str = "baseline", nested: bool = False) -> dict:
    """Registra un run de P1; ``modelo`` es una clave de ``HIPERPARAMS_POR_DEFECTO["p1"]``."""
    train, test = datos
    if modelo == "naive":
        columna = hiperparams["columna_naive"]
        return ejecutar_run(
            f"P1-baseline-naive-{columna.removeprefix('lag_salidas_')}",
            TIPOS_NAIVE.get(columna, "naive"), train, test, params_base,
            columna_naive=columna, stage=stage, nested=nested,
        )
    else:
        return ejecutar_run(
            "P1-baseline-histgb", "HistGradientBoostingRegressor", train, test, params_base,
            modelo=HistGradientBoostingRegressor(**hiperparams, random_state=RANDOM_STATE),
            stage=stage, nested=nested,
        )


def main():
    datos = cargar_datos()
    configurar_experimento()
    params_base = params_reproducibilidad()

    defaults = HIPERPARAMS_POR_DEFECTO["p1"]
    entrenar("naive", {"columna_naive": "lag_salidas_1h"}, datos, params_base)
    entrenar("naive", defaults["naive"], datos, params_base)
    entrenar("histgb", defaults["histgb"], datos, params_base)


if __name__ == "__main__":
    main()
