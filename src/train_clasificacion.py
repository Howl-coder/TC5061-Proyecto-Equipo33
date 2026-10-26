"""P2 (Clasificación): baselines de riesgo de desbalance crítico por estación, registrados en MLflow.

Uso: python -m src.train_clasificacion
"""

import matplotlib

matplotlib.use("Agg")
from config import (
    BASE_DIR,
    FECHA_INICIO_TEST,
    FIGURES_DIR,
    HIPERPARAMS_POR_DEFECTO,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RUTA_CATALOGO_ESTACIONES,
    RUTA_MANIFIESTO_DATOS,
)
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import cargar_catalogo_amg
from src.features import agregar_features_y_target_desbalance
from src.tracking import (
    SKOPS_TIPOS_CONFIABLES,
    configurar_experimento,
    params_reproducibilidad,
    tags_run,
)

TARGET = "en_riesgo_desbalance"
HORIZONTE_H = 3
UMBRAL_FRACCION_CAPACIDAD = 0.3
UMBRAL_DECISION = 0.5
FEATURES = [
    "hora_sin",
    "hora_cos",
    "dia_sin",
    "dia_cos",
    "es_fin_de_semana",
    "dpcapacity",
    "balance_neto_h",
    "lag_balance_1h",
    "lag_balance_2h",
    "balance_acum_3h",
    "demanda_salidas",
    "flujo_llegadas",
    "lag_salidas_1h",
    "lag_salidas_24h",
    "lag_salidas_168h",
    "rolling_mean_6h",
]
MODELOS = {
    "logreg": (
        "LogisticRegression",
        lambda hp: make_pipeline(
            StandardScaler(), LogisticRegression(**hp, random_state=RANDOM_STATE)
        ),
    ),
    "histgb": (
        "HistGradientBoostingClassifier",
        lambda hp: HistGradientBoostingClassifier(**hp, random_state=RANDOM_STATE),
    ),
}


def metricas_clasificacion(y_true, y_proba) -> dict:
    y_pred = (y_proba >= UMBRAL_DECISION).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def graficar_evaluacion(y_true, y_proba, nombre: str, metric_prefix: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[0], plot_chance_level=True)
    axes[0].set_title("Curva ROC")
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=axes[1], plot_chance_level=True)
    axes[1].set_title("Curva Precision-Recall")
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        (y_proba >= UMBRAL_DECISION).astype(int),
        ax=axes[2],
        colorbar=False,
        display_labels=["Normal", "Riesgo"],
    )
    axes[2].set_title(f"Matriz de confusión (umbral {UMBRAL_DECISION})")
    fig.suptitle(f"P2 {nombre} — {metric_prefix}")
    return fig


def ejecutar_run(
    nombre_run,
    tipo_modelo,
    modelo,
    hiperparams,
    train,
    test,
    params_base,
    stage="baseline",
    nested=False,
    metric_prefix="test",
) -> dict:
    with mlflow.start_run(run_name=nombre_run, nested=nested):
        mlflow.set_tags(tags_run("P2", tipo_modelo, stage=stage))
        mlflow.log_params(params_base)
        mlflow.log_params(
            {
                "target": TARGET,
                "horizonte_h": HORIZONTE_H,
                "umbral_fraccion_capacidad": UMBRAL_FRACCION_CAPACIDAD,
                "umbral_decision": UMBRAL_DECISION,
                "features": ",".join(FEATURES),
                "n_train": len(train),
                f"n_{metric_prefix}": len(test),
                "n_estaciones": test["Origen_Id"].nunique(),
            }
        )
        mlflow.log_params(hiperparams)
        mlflow.log_metrics(
            {
                "tasa_positivos_train": float(train[TARGET].mean()),
                f"tasa_positivos_{metric_prefix}": float(test[TARGET].mean()),
            }
        )

        modelo.fit(train[FEATURES], train[TARGET])
        y_proba = modelo.predict_proba(test[FEATURES])[:, 1]
        metricas = {
            f"{metric_prefix}_{k}": v
            for k, v in metricas_clasificacion(test[TARGET], y_proba).items()
        }
        mlflow.log_metrics(metricas)

        signature = infer_signature(train[FEATURES], modelo.predict_proba(train[FEATURES]))
        mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            signature=signature,
            input_example=train[FEATURES].iloc[:3],
            skops_trusted_types=SKOPS_TIPOS_CONFIABLES,
        )

        dataset_train = mlflow.data.from_pandas(
            train[FEATURES + [TARGET]],
            source=str(PROCESSED_DATA_DIR / "features_demanda_p1.parquet"),
            name="mibici-train-p2",
            targets=TARGET,
        )
        mlflow.log_input(dataset_train, context="training")

        fig = graficar_evaluacion(test[TARGET], y_proba, nombre_run, metric_prefix)
        ruta_fig = FIGURES_DIR / f"{nombre_run}_evaluacion.png"
        fig.savefig(ruta_fig, dpi=120, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(ruta_fig), artifact_path="figuras")

        mlflow.log_artifact(str(RUTA_MANIFIESTO_DATOS), artifact_path="datos")
        mlflow.log_artifact(str(BASE_DIR / "config.py"), artifact_path="config")

        print(f"{nombre_run}: {metricas}")
    return metricas


def cargar_datos():
    """Devuelve ``(train, test)`` con el target de P2 y la partición temporal de ``config``."""
    ruta = PROCESSED_DATA_DIR / "features_demanda_p1.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"Primero corre src.dataset y src.features para generar {ruta}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = agregar_features_y_target_desbalance(
        pd.read_parquet(ruta),
        cargar_catalogo_amg(RUTA_CATALOGO_ESTACIONES),
        HORIZONTE_H,
        UMBRAL_FRACCION_CAPACIDAD,
    )
    corte = pd.Timestamp(FECHA_INICIO_TEST)
    corte_entrenamiento = corte - pd.Timedelta(hours=HORIZONTE_H)
    return (
        df[df["timestamp_hora"] < corte_entrenamiento],
        df[df["timestamp_hora"] >= corte],
    )


def entrenar(
    modelo: str,
    hiperparams: dict,
    datos,
    params_base: dict,
    stage: str = "baseline",
    nested: bool = False,
) -> dict:
    """Registra un run de P2; ``modelo`` es una clave de ``MODELOS``."""
    train, test = datos
    tipo_modelo, construir = MODELOS[modelo]
    return ejecutar_run(
        f"P2-{stage}-{modelo}",
        tipo_modelo,
        construir(hiperparams),
        hiperparams,
        train,
        test,
        params_base,
        stage=stage,
        nested=nested,
    )


def main():
    datos = cargar_datos()
    configurar_experimento()
    params_base = params_reproducibilidad()

    for modelo, hiperparams in HIPERPARAMS_POR_DEFECTO["p2"].items():
        entrenar(modelo, hiperparams, datos, params_base)


if __name__ == "__main__":
    main()
