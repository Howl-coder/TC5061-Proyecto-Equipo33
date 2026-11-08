# Parte C — Model Registry y reporte

Promoción registrada: 2026-11-08 20:40 UTC.

Se usa el alias MLflow `champion` en vez de las etapas legacy (`Production`), porque los aliases son el mecanismo vigente y permiten mover el nombre de producción sin crear otra versión del modelo.

La promoción registra cada modelo desde su run de origen; los tags de la versión conservan experimento, run, commit, etapa y versión de datos. La duración de entrenamiento se reporta como referencia de costo; el costo de inferencia no se midió en esta entrega.

## P1 — Pronóstico de demanda por estación

- **Modelo registrado:** `mibici-equipo33-p1-demanda` versión `1`.
- **Alias de producción:** `mibici-equipo33-p1-demanda@champion`.
- **Run de origen:** `606849422b6941c38df8c9039bdcff5d` (`P1-baseline-histgb`).
- **Familia / etapa:** `HistGradientBoostingRegressor` / `baseline`.
- **Linaje:** experimento `mibici-equipo33`, commit `c59e6e0c7fa67f9db806ee5a2ec2eff3cd185695`, datos `mibici-datos-abiertos-2026_04-2026_09`.
- **Duración de entrenamiento:** 86.6 s.
- **Justificación de selección:** HistGradientBoosting con RMSE 1.481, MAE 0.837 y R2 0.692 en septiembre; mejora clara sobre los baselines naive de persistencia y estacional.

| Run | Etapa | Familia | test_rmse | test_mae | test_r2 | Entrenamiento (s) |
|---|---|---|---|---|---|---:|
| `P1-baseline-histgb` **(seleccionado)** | baseline | HistGradientBoostingRegressor | 1.4810 | 0.8373 | 0.6917 | 86.6 |
| `P1-baseline-histgb` | baseline | HistGradientBoostingRegressor | 1.4810 | 0.8373 | 0.6917 | 388.5 |

## P2 — Riesgo de desbalance de estaciones

- **Modelo registrado:** `mibici-equipo33-p2-riesgo` versión `1`.
- **Alias de producción:** `mibici-equipo33-p2-riesgo@champion`.
- **Run de origen:** `7f7d3a0333324db9bc5434f6d6d72912` (`P2-baseline-histgb`).
- **Familia / etapa:** `HistGradientBoostingClassifier` / `baseline`.
- **Linaje:** experimento `mibici-equipo33`, commit `c59e6e0c7fa67f9db806ee5a2ec2eff3cd185695`, datos `mibici-datos-abiertos-2026_04-2026_09`.
- **Duración de entrenamiento:** 94.2 s.
- **Justificación de selección:** HistGradientBoosting con recall 0.881 y PR-AUC 0.339 en septiembre, entrenado con la purga de tres horas; se prioriza recall por el costo de no detectar estaciones en riesgo.

| Run | Etapa | Familia | test_recall | test_pr_auc | test_f1 | test_roc_auc | Entrenamiento (s) |
|---|---|---|---|---|---|---|---:|
| `P2-baseline-histgb` **(seleccionado)** | baseline | HistGradientBoostingClassifier | 0.8810 | 0.3391 | 0.3207 | 0.8563 | 94.2 |

## P3 — Segmentación de cicloestaciones

- **Modelo registrado:** `mibici-equipo33-p3-grupos` versión `1`.
- **Alias de producción:** `mibici-equipo33-p3-grupos@champion`.
- **Run de origen:** `6df3267dbe25412188e7a787b30566a2` (`P3-baseline-kmeans`).
- **Familia / etapa:** `KMeans` / `baseline`.
- **Linaje:** experimento `mibici-equipo33`, commit `c59e6e0c7fa67f9db806ee5a2ec2eff3cd185695`, datos `mibici-datos-abiertos-2026_04-2026_09`.
- **Duración de entrenamiento:** 29.3 s.
- **Justificación de selección:** K-Means con k=3 elegido por silhouette (0.095) y Davies-Bouldin 2.61 sobre perfiles del periodo de entrenamiento.

| Run | Etapa | Familia | silhouette | davies_bouldin | calinski_harabasz | Entrenamiento (s) |
|---|---|---|---|---|---|---:|
| `P3-baseline-kmeans` **(seleccionado)** | baseline | KMeans | 0.0948 | 2.6091 | 39.2670 | 29.3 |

## Verificación

Los tres modelos se cargaron por su URI de versión y, después de asignar el alias, se volvieron a cargar mediante `models:/<nombre>@champion`. No se registran datos personales de usuarios de MiBici.
