"""P3 (Clustering): baseline K-Means de perfiles de uso por estación, registrado en MLflow.

Uso: python -m src.train_clustering
"""

import matplotlib

matplotlib.use("Agg")
from config import (
    BASE_DIR,
    FECHA_FIN_TRAIN,
    FIGURES_DIR,
    HIPERPARAMS_POR_DEFECTO,
    INTERIM_DATA_DIR,
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
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import cargar_catalogo_amg
from src.features import construir_perfil_estaciones
from src.tracking import (
    SKOPS_TIPOS_CONFIABLES,
    configurar_experimento,
    params_reproducibilidad,
    tags_run,
)

FEATURES = [f"prop_hora_{h:02d}" for h in range(24)] + [
    "prop_fin_de_semana",
    "prop_salidas",
    "prop_circulares",
    "viajes_por_dia",
    "uso_por_capacidad",
]


def graficar_mapa(perfil: pd.DataFrame, k: int):
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        perfil["longitude"],
        perfil["latitude"],
        c=perfil["cluster_estacion"],
        cmap="tab10",
        vmin=0,
        vmax=9,
        s=12 + 4 * perfil["viajes_por_dia"] ** 0.5,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.3,
    )
    ax.legend(*sc.legend_elements(), title="Cluster", loc="lower left")
    ax.set_title(f"P3 — Grupos de cicloestaciones MiBici (K-Means, k={k})")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal")
    return fig


def graficar_perfiles_horarios(perfil: pd.DataFrame):
    cols = [f"prop_hora_{h:02d}" for h in range(24)]
    medias = perfil.groupby("cluster_estacion")[cols].mean()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for cluster, fila in medias.iterrows():
        n = (perfil["cluster_estacion"] == cluster).sum()
        ax.plot(range(24), fila.values, marker="o", ms=3, label=f"Cluster {cluster} (n={n})")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Proporción de salidas")
    ax.set_title("P3 — Perfil horario promedio por cluster")
    ax.legend()
    return fig


def graficar_curva_k(resultados: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(resultados["k"], resultados["silhouette"], marker="o", color="tab:blue")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(resultados["k"], resultados["davies_bouldin"], marker="s", color="tab:red")
    ax2.set_ylabel("Davies-Bouldin", color="tab:red")
    ax1.set_title("P3 — Selección de k")
    return fig


def guardar_y_registrar(fig, nombre: str):
    ruta = FIGURES_DIR / nombre
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(ruta), artifact_path="figuras")


def cargar_datos() -> pd.DataFrame:
    """Devuelve el perfil de cada estación calculado solo con el periodo de entrenamiento."""
    ruta = INTERIM_DATA_DIR / "viajes_amg_limpios.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"Primero corre src.dataset para generar {ruta}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    return construir_perfil_estaciones(
        pd.read_parquet(ruta), cargar_catalogo_amg(RUTA_CATALOGO_ESTACIONES), FECHA_FIN_TRAIN
    )


def entrenar(
    nombre_modelo: str,
    hiperparams: dict,
    perfil: pd.DataFrame,
    params_base: dict,
    stage: str = "baseline",
    nested: bool = False,
) -> dict:
    """Registra el run de P3: elige k en ``[k_min, k_max]`` por silhouette y ajusta K-Means."""
    k_min, k_max, n_init = hiperparams["k_min"], hiperparams["k_max"], hiperparams["n_init"]
    X = StandardScaler().fit_transform(perfil[FEATURES])

    with mlflow.start_run(run_name=f"P3-{stage}-{nombre_modelo}", nested=nested):
        mlflow.set_tags(tags_run("P3", "KMeans", stage=stage))
        mlflow.log_params(params_base)
        mlflow.log_params(
            {
                "periodo_perfil": f"fecha_inicio..{FECHA_FIN_TRAIN}",
                "features": ",".join(FEATURES),
                "escalado": "StandardScaler",
                "k_min": k_min,
                "k_max": k_max,
                "n_init": n_init,
                "criterio_seleccion_k": "max_silhouette",
                "n_estaciones": len(perfil),
            }
        )

        filas = []
        for k in range(k_min, k_max + 1):
            etiquetas = KMeans(n_clusters=k, n_init=n_init, random_state=RANDOM_STATE).fit_predict(
                X
            )
            fila = {
                "k": k,
                "silhouette": silhouette_score(X, etiquetas),
                "davies_bouldin": davies_bouldin_score(X, etiquetas),
            }
            filas.append(fila)
            mlflow.log_metrics(
                {
                    "silhouette_por_k": fila["silhouette"],
                    "davies_bouldin_por_k": fila["davies_bouldin"],
                },
                step=k,
            )
        resultados = pd.DataFrame(filas)
        k_opt = int(resultados.loc[resultados["silhouette"].idxmax(), "k"])

        pipeline = make_pipeline(
            StandardScaler(), KMeans(n_clusters=k_opt, n_init=n_init, random_state=RANDOM_STATE)
        )
        perfil = perfil.copy()
        perfil["cluster_estacion"] = pipeline.fit_predict(perfil[FEATURES])
        X_opt = pipeline[:-1].transform(perfil[FEATURES])
        etiquetas = perfil["cluster_estacion"]
        metricas = {
            "silhouette": float(silhouette_score(X_opt, etiquetas)),
            "davies_bouldin": float(davies_bouldin_score(X_opt, etiquetas)),
            "calinski_harabasz": float(calinski_harabasz_score(X_opt, etiquetas)),
            "inercia": float(pipeline[-1].inertia_),
            "tamano_cluster_min": int(etiquetas.value_counts().min()),
            "tamano_cluster_max": int(etiquetas.value_counts().max()),
        }
        mlflow.log_param("k", k_opt)
        mlflow.log_metrics(metricas)

        signature = infer_signature(perfil[FEATURES], pipeline.predict(perfil[FEATURES]))
        mlflow.sklearn.log_model(
            pipeline,
            name="modelo",
            signature=signature,
            input_example=perfil[FEATURES].iloc[:3],
            skops_trusted_types=SKOPS_TIPOS_CONFIABLES,
        )

        dataset_perfil = mlflow.data.from_pandas(
            perfil[FEATURES],
            source=str(INTERIM_DATA_DIR / "viajes_amg_limpios.parquet"),
            name="mibici-perfiles-p3",
        )
        mlflow.log_input(dataset_perfil, context="training")

        guardar_y_registrar(graficar_mapa(perfil, k_opt), "P3_mapa_clusters.png")
        guardar_y_registrar(graficar_perfiles_horarios(perfil), "P3_perfiles_horarios.png")
        guardar_y_registrar(graficar_curva_k(resultados), "P3_seleccion_k.png")

        asignacion = perfil[
            [
                "id",
                "name",
                "latitude",
                "longitude",
                "dpcapacity",
                "viajes_por_dia",
                "cluster_estacion",
            ]
        ]
        mlflow.log_table(asignacion, artifact_file="tablas/asignacion_clusters.json")
        mlflow.log_table(resultados, artifact_file="tablas/seleccion_k.json")

        mlflow.log_artifact(str(RUTA_MANIFIESTO_DATOS), artifact_path="datos")
        mlflow.log_artifact(str(BASE_DIR / "config.py"), artifact_path="config")

        print(f"P3 k={k_opt}: {metricas}")
    return metricas


def main():
    perfil = cargar_datos()
    configurar_experimento()
    entrenar("kmeans", HIPERPARAMS_POR_DEFECTO["p3"]["kmeans"], perfil, params_reproducibilidad())


if __name__ == "__main__":
    main()
