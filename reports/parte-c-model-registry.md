# Parte C — Model Registry y reporte

## Estado

La promoción debe ejecutarse después de lanzar y comparar los runs de la Parte B en el
experimento `mibici-equipo33`. Este repositorio no incluye el backend local `mlflow.db` ni
registra métricas de experimentos ejecutados; por ello, aquí no se inventan IDs de runs,
versiones registradas ni resultados de candidatos.

Los resultados baseline publicados en el README son referencias iniciales, no una
selección de producción para la Parte C. La decisión definitiva se genera con los
resultados reales al ejecutar `src.register_models`.

## Criterios de selección

La selección se realiza en la interfaz de MLflow, conservando la partición temporal y
comparando baselines y candidatos del mismo experimento:

| Pregunta | Métrica principal | Métricas complementarias | Costo considerado |
|---|---|---|---|
| P1 — demanda | `test_rmse` (menor es mejor) | `test_mae`, `test_r2` | Duración medida del entrenamiento |
| P2 — riesgo | `test_recall` (mayor sensibilidad ante estaciones en riesgo) | `test_pr_auc`, `test_f1`, `test_roc_auc` | Duración medida del entrenamiento |
| P3 — grupos | `silhouette` (mayor separación/cohesión) | `davies_bouldin` (menor es mejor), `calinski_harabasz` | Duración medida del entrenamiento |

La duración es solo una aproximación al costo computacional; no se midió el costo de
inferencia. La justificación concreta de cada run seleccionado se proporciona al
registrarlo y se conserva en la descripción de la versión y en el reporte generado.

Para no usar el test durante la optimización, Optuna ajusta P1 y P2 con abril-junio y
valida en julio; septiembre se reserva para la evaluación final de baselines y candidatos
reentrenados con el periodo de entrenamiento. P2 excluye tres horas en los límites de
entrenamiento por el horizonte de su target. P3 calcula perfiles solo con viajes del
periodo de entrenamiento y compara silhouette/Davies-Bouldin sin consultar septiembre.

## Registro y ciclo de vida

Se usa el alias MLflow `champion`, en vez de las etapas legacy (`Production`), porque los
aliases son el esquema vigente. Los nombres estables son:

- `mibici-equipo33-p1-demanda`
- `mibici-equipo33-p2-riesgo`
- `mibici-equipo33-p3-grupos`

El script `src.register_models` exige runs finalizados del experimento correcto, la
pregunta correspondiente, árbol Git limpio, commit, `data_version`, métricas requeridas
y una firma cargable. Registra cada modelo desde `runs:/<run_id>/modelo`, anota el linaje
en tags de la versión y comprueba que cada modelo se carga desde la versión y desde
`models:/<nombre>@champion`. Al completar la promoción, el script reemplaza este archivo
con la comparación real y las justificaciones indicadas.

Ejemplo (reemplaza cada ID y justificación por la decisión tomada tras revisar MLflow):

```powershell
python -m src.register_models `
  --p1-run-id <RUN_ID_P1> --p1-justification "Mejor RMSE que el baseline; MAE y R2 consistentes." `
  --p2-run-id <RUN_ID_P2> --p2-justification "Recall priorizado para reducir estaciones de riesgo no detectadas." `
  --p3-run-id <RUN_ID_P3> --p3-justification "Silhouette y Davies-Bouldin favorecen grupos útiles y estables."
```

Para verificar una vez publicado:

```powershell
python -c "import mlflow.pyfunc; mlflow.pyfunc.load_model('models:/mibici-equipo33-p1-demanda@champion')"
python -c "import mlflow.pyfunc; mlflow.pyfunc.load_model('models:/mibici-equipo33-p2-riesgo@champion')"
python -c "import mlflow.pyfunc; mlflow.pyfunc.load_model('models:/mibici-equipo33-p3-grupos@champion')"
```
