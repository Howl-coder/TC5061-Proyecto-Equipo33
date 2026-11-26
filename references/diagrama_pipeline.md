# Diagrama del pipeline — MiBici Equipo 33

```mermaid
flowchart TD
    A["CSV mensuales de viajes 2026-04 a 2026-09<br/>+ nomenclatura_2026_09.csv<br/>(data/raw/)"] --> V{"Verificación SHA-256<br/>contra references/manifiesto_datos.json"}
    V -- "no coincide" --> X["Error: versión de datos distinta<br/>(--actualizar-manifiesto si es intencional)"]
    V -- "coincide" --> B["src/dataset.py<br/>homologa columnas y codificación, deduplica Viaje_Id,<br/>filtra fechas, duración 1 min–24 h y estaciones del AMG"]
    B --> C[("data/interim/<br/>viajes_amg_limpios.parquet")]

    C --> D["src/features.py<br/>malla hora × estación de origen o destino con ceros,<br/>variables cíclicas, rezagos 1/2/3/24/168 h, media móvil 6 h"]
    D --> E[("data/processed/<br/>features_demanda_p1.parquet")]

    E --> P1["P1 · src/train_regresion.py<br/>target: demanda_salidas en la hora t<br/>naive lag 1 h · naive lag 168 h · HistGradientBoostingRegressor"]
    E --> F2["features.agregar_features_y_target_desbalance<br/>capacidad, rezagos de balance,<br/>target en_riesgo_desbalance (3 h, 30 % capacidad)"]
    F2 --> P2["P2 · src/train_clasificacion.py<br/>LogisticRegression · HistGradientBoostingClassifier<br/>(class_weight=balanced)"]
    C --> F3["features.construir_perfil_estaciones<br/>perfil horario y de uso por estación<br/>(solo periodo de entrenamiento)"]
    F3 --> P3["P3 · src/train_clustering.py<br/>StandardScaler + K-Means, k elegido por silhouette"]

    P1 --> M[("MLflow · experimento mibici-equipo33<br/>params: fechas, data_version, data_sha256, hiperparámetros<br/>tags: equipo, pregunta, git_commit, git_dirty<br/>métricas + figuras + modelo")]
    P2 --> M
    P3 --> M
```

**Partición temporal (P1 y P2):** entrenamiento del 2026-04-01 al 2026-08-31, prueba del 2026-09-01 al 2026-09-30.

**Métricas registradas:**

| Pregunta | Métricas |
|---|---|
| P1 | RMSE, MAE, R² (test y train) |
| P2 | ROC-AUC, PR-AUC, Precision, Recall, F1 (clase en riesgo) |
| P3 | Silhouette, Davies-Bouldin, Calinski-Harabasz, inercia, tamaño de clusters |
