# MiBici Mobility Analytics & MLOps Pipeline (Equipo 33)

Pipeline reproducible de Machine Learning para pronosticar la demanda, detectar el riesgo de desbalance y segmentar las cicloestaciones del sistema de bicicletas públicas MiBici en el Área Metropolitana de Guadalajara (AMG).

## Equipo 33

| Integrante | Rol |
|---|---|
| Leonardo Contreras Palomino | Análisis exploratorio y validación de modelos |
| Sofía Donlucas Bañuelos | Ingesta de datos y reproducibilidad |
| Gerardo Duarte Ceballos | Configuración del repositorio e integración |
| Alejandro Gutiérrez Saldaña | Feature engineering y experimentos MLflow |

## Preguntas del proyecto

| | Tipo | Pregunta | Target |
|---|---|---|---|
| **P1** | Regresión | ¿Cuántos viajes iniciarán en una estación en la próxima hora? | `demanda_salidas` |
| **P2** | Clasificación | ¿Una estación estará en riesgo de quedarse sin bicis o sin anclajes en las próximas horas? | `en_riesgo_desbalance` |
| **P3** | Clustering | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | `cluster_estacion` |

El planteamiento completo está en el [ML Canvas](references/ml_canvas_v1.2.docx) y en la [definición de datos y pipeline](references/definicion_datos_y_pipeline.docx). El flujo de punta a punta está en el [diagrama del pipeline](references/diagrama_pipeline.md).

## Datos

- **Fuente:** [Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/), viajes mensuales y nomenclatura de estaciones.
- **Rango:** 2026-04-01 a 2026-09-30 (6 meses).
- **Partición temporal:** entrenamiento de abril a agosto y prueba en septiembre. No se usa partición aleatoria, para no filtrar información del futuro.
- **Archivos esperados en `data/raw/`:** `datos_abiertos_2026_04.csv` … `datos_abiertos_2026_09.csv` y `nomenclatura_2026_09.csv`. Se descargan del portal y **no se versionan en Git**.

### Versionado de datos

[references/manifiesto_datos.json](references/manifiesto_datos.json), que sí está en Git, guarda el tamaño y el SHA-256 de cada archivo crudo y un hash global de la versión (`data_version` = `mibici-datos-abiertos-2026_04-2026_09`).

- `python -m src.dataset` compara los CSV locales contra el manifiesto y **se detiene si no coinciden**, indicando qué archivo cambió.
- Cada run de MLflow verifica los datos otra vez y registra `data_version` y `data_sha256`.
- Para registrar una versión nueva a propósito, por ejemplo al agregar un mes, ejecuta `python -m src.dataset --actualizar-manifiesto` y haz commit del manifiesto.

## Estructura del repositorio

Generada con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) (`ccds`, `module_name = src`).

```text
├── Makefile               # make requirements | lint | format | clean
├── MLproject              # MLflow Project: entry points ingest, features, train y search
├── README.md
├── config.py              # Rutas, fechas, filtros AMG, MLflow y modelos/hiperparámetros por defecto
├── pyproject.toml         # Metadatos del proyecto y configuración de ruff
├── python_env.yaml        # Entorno del MLproject (mismas versiones que requirements.txt)
├── requirements.txt       # Dependencias con versión fija (Python 3.12)
├── data/                  # No versionado
│   ├── raw/               # CSV originales de MiBici
│   ├── interim/           # viajes_amg_limpios.parquet
│   └── processed/         # features_demanda_p1.parquet
├── docs/
├── models/                # Modelos locales (no versionados; los oficiales viven en MLflow)
├── notebooks/
│   └── 1.0-lcp-exploratorio.ipynb   # Análisis exploratorio (EDA)
├── references/
│   ├── ml_canvas_v1.2.docx
│   ├── definicion_datos_y_pipeline.docx
│   ├── diagrama_pipeline.md
│   └── manifiesto_datos.json        # Versión de los datos crudos (SHA-256)
├── reports/
│   ├── figures/           # Figuras generadas (también se guardan como artefactos en MLflow)
│   └── parte-c-model-registry.md  # Comparación y justificación de modelos promovidos
└── src/
    ├── dataset.py         # Verificación de versión, limpieza y filtrado AMG
    ├── features.py        # Malla horaria, variables cíclicas, rezagos, targets P2 y perfiles P3
    ├── tracking.py        # Utilidades comunes de MLflow (tags, parámetros, verificación de datos)
    ├── train.py           # Entrena un modelo (pregunta, modelo, hiperparámetros); usado por MLproject
    ├── search.py          # Búsqueda Optuna con runs anidados y validación temporal
    ├── register_models.py # Registra los modelos elegidos y promueve el alias champion
    ├── train_regresion.py       # Baselines P1
    ├── train_clasificacion.py   # Baselines P2
    └── train_clustering.py      # Baseline P3
```

## Reproducción desde un clon limpio

Todas las corridas se lanzan desde el [MLproject](MLproject) con `mlflow run`, no desde notebooks. Se requiere Python 3.12 y [uv](https://docs.astral.sh/uv/). MLflow usa `uv` para crear el entorno declarado en [python_env.yaml](python_env.yaml). Los comandos son para PowerShell.

**1. Clonar e instalar el CLI de MLflow**

```powershell
git clone https://github.com/Howl-coder/TC5061-Proyecto-Equipo33.git
cd TC5061-Proyecto-Equipo33
python -m venv .venv; .\.venv\Scripts\Activate.ps1   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**2. Levantar el servidor de tracking** (terminal aparte, desde la raíz del repo; debe quedar abierta)

El Model Registry requiere un backend con base de datos. `mlflow.db` se crea en la primera ejecución y no se versiona.

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

**3. Ingesta y features**

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"   # bash: export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

mlflow run . -e ingest   --experiment-name mibici-equipo33 --env-manager uv   # descarga CSVs, verifica el manifiesto y limpia -> data/interim/
mlflow run . -e features --experiment-name mibici-equipo33 --env-manager uv   # malla horaria y variables -> data/processed/
```

**4. Modelo elegido de cada pregunta** (hiperparámetros por defecto de `config.py`)

```powershell
mlflow run . -e train -P pregunta=p1 -P modelo=histgb --experiment-name mibici-equipo33 --env-manager uv
mlflow run . -e train -P pregunta=p2 -P modelo=histgb --experiment-name mibici-equipo33 --env-manager uv
mlflow run . -e train -P pregunta=p3 -P modelo=kmeans --experiment-name mibici-equipo33 --env-manager uv
```

**5. Búsqueda de hiperparámetros**

```powershell
mlflow run . -e search -P pregunta=p1 --experiment-name mibici-equipo33 --env-manager uv
mlflow run . -e search -P pregunta=p2 --experiment-name mibici-equipo33 --env-manager uv
mlflow run . -e search -P pregunta=p3 --experiment-name mibici-equipo33 --env-manager uv
```

Las búsquedas supervisadas P1 y P2 ajustan sus combinaciones con datos de abril a junio
y validan en julio. En P2 se excluyen las tres horas anteriores al corte para evitar que
las etiquetas de riesgo del entrenamiento incluyan balances de la validación. Septiembre
permanece reservado para la comparación final y no se usa en Optuna. P3 calcula perfiles
solo hasta agosto, como en el baseline, y compara silhouette dentro de esos perfiles sin
usar datos de septiembre.

Tras escoger los hiperparámetros con las métricas `best_validation_*` del run padre (P3
elige por silhouette usando solo los perfiles de entrenamiento), vuelve a ejecutar cada
configuración como `candidate`: el entry point `train` ajusta con todo el periodo de
entrenamiento y evalúa una sola vez en septiembre. Sustituye los hiperparámetros por los
ganadores observados en MLflow. En P2, usa `tipo_modelo` del run padre para elegir
`-P modelo=logreg` o `-P modelo=histgb`; no lo pases como hiperparámetro:

```powershell
mlflow run . -e train -P pregunta=p1 -P modelo=histgb -P hiperparams="max_iter=300,learning_rate=0.05,max_depth=5,min_samples_leaf=40,l2_regularization=0.1" -P stage=candidate --experiment-name mibici-equipo33 --env-manager uv
mlflow run . -e train -P pregunta=p2 -P modelo=histgb -P hiperparams="max_iter=300,learning_rate=0.05,max_depth=5" -P stage=candidate --experiment-name mibici-equipo33 --env-manager uv
mlflow run . -e train -P pregunta=p3 -P modelo=kmeans -P hiperparams="k_min=3,k_max=10,n_init=10" -P stage=candidate --experiment-name mibici-equipo33 --env-manager uv
```

En P2 el entrenamiento final también deja un margen de tres horas antes del test, pues
el target observa el balance de las siguientes tres horas. Los baselines de P2 deben
volverse a correr con esta purga antes de comparar sus métricas con las de los candidatos.

| Pregunta | Modelo elegido | Run | Hiperparámetros (`config.HIPERPARAMS_POR_DEFECTO`) |
|---|---|---|---|
| P1 | `HistGradientBoostingRegressor` | `P1-baseline-histgb` | `max_iter=200`, `learning_rate=0.1`, `max_depth=None`, `min_samples_leaf=20`, `l2_regularization=0.0` |
| P2 | `HistGradientBoostingClassifier` | `P2-baseline-histgb` | `max_iter=200`, `learning_rate=0.1`, `max_depth=None`, `class_weight=balanced` |
| P3 | `StandardScaler` + `KMeans` | `P3-baseline-kmeans` | `k_min=3`, `k_max=10`, `n_init=10` (k elegido por silhouette) |

Los baselines de comparación se lanzan igual, por ejemplo:

```powershell
mlflow run . -e train -P pregunta=p1 -P modelo=naive -P hiperparams="columna_naive=lag_salidas_1h" --experiment-name mibici-equipo33 --env-manager uv
mlflow run . -e train -P pregunta=p1 -P modelo=naive --experiment-name mibici-equipo33 --env-manager uv
mlflow run . -e train -P pregunta=p2 -P modelo=logreg --experiment-name mibici-equipo33 --env-manager uv
```

La interfaz de MLflow queda disponible en http://127.0.0.1:5000.

### Entry points del MLproject

| Entry point | Parámetros | Ejecuta |
|---|---|---|
| `ingest` | — | `python -m src.dataset` |
| `features` | — | `python -m src.features` |
| `train` | `pregunta` (`p1`, `p2`, `p3`; obligatorio), `modelo`, `hiperparams`, `stage` (`baseline` o `candidate`) | `python -m src.train` (un run por ejecución) |
| `search` | `pregunta` (`p1`, `p2`, `p3`; obligatorio), `n_trials` (default: 5) | `python -m src.search` (run padre y runs hijos anidados) |

- `modelo` e `hiperparams` valen `config` por defecto: se toman `MODELOS_POR_DEFECTO` y `HIPERPARAMS_POR_DEFECTO` de [config.py](config.py). Rutas, fechas, partición y semilla también salen de ahí.
- Modelos disponibles: P1 `naive`, `histgb`; P2 `logreg`, `histgb`; P3 `kmeans`.
- `hiperparams` usa el formato `clave=valor,clave=valor`, sin espacios, y solo reemplaza las claves indicadas, por ejemplo `-P hiperparams="max_iter=300,learning_rate=0.05"`. Las claves válidas son las de `HIPERPARAMS_POR_DEFECTO` para ese modelo.
- `--experiment-name mibici-equipo33` es obligatorio: el script fija ese experimento y MLflow rechaza el run si no coincide. Si el experimento no existe, MLflow lo crea.
- Con `--env-manager local` se reutiliza el entorno ya activado en lugar de crear uno con `uv`.
- Para otro servidor de tracking, cambia `MLFLOW_TRACKING_URI`; `config.py` usa `http://127.0.0.1:5000` si la variable no está definida.

> Para que los runs queden ligados a un commit exacto, lánzalos con el árbol de trabajo limpio (`git status` sin cambios). Si hay cambios sin commit, el run se etiqueta con `git_dirty=True`.

## Experimento en MLflow

- **Experimento:** `mibici-equipo33`, registrado en el servidor local `http://127.0.0.1:5000`, respaldado por `mlflow.db` (SQLite, no versionado).
- **Tags de cada run:** `equipo`, `question` (`P1`/`P2`/`P3`), `git_commit`, `git_dirty`, `model_family`, `stage_of_work` (`baseline`, `search`, `candidate`).
- **Parámetros de cada run:** `fecha_inicio`, `fecha_fin`, `fecha_fin_train`, `fecha_inicio_test`, `data_version`, `data_sha256`, archivos usados, `random_state` e hiperparámetros del modelo.
- **Artefactos:** figura de evaluación de cada run y el modelo serializado. P3 agrega además el mapa de clusters, los perfiles horarios, la curva de selección de k y la tabla de asignación de estaciones.

| Run | Pregunta | Modelo |
|---|---|---|
| `P1-baseline-naive-1h` | P1 | Persistencia: misma demanda que la hora anterior |
| `P1-baseline-naive-168h` | P1 | Naive estacional: misma hora de la semana anterior |
| `P1-baseline-histgb` | P1 | `HistGradientBoostingRegressor` |
| `P2-baseline-logreg` | P2 | `LogisticRegression` (`class_weight=balanced`) |
| `P2-baseline-histgb` | P2 | `HistGradientBoostingClassifier` (`class_weight=balanced`) |
| `P3-baseline-kmeans` | P3 | `StandardScaler` + `KMeans`, k ∈ [3, 10] elegido por silhouette |

Los runs registrados antes del MLproject conservan sus nombres anteriores (`P1-baseline-naive-lag1h`, `P2-baseline-logistic-regression`, etc.).

### Resultados baseline (test: septiembre 2026)

| Pregunta | Mejor baseline | Métricas |
|---|---|---|
| P1 | HistGradientBoosting | RMSE 1.48 · MAE 0.84 · R² 0.69 (naive 168 h: RMSE 2.00 · R² 0.44) |
| P2 | HistGradientBoosting | ROC-AUC 0.86 · PR-AUC 0.34 · Recall 0.88 · F1 0.32 (positivos ≈ 6.6 %; referencia previa a la purga temporal) |
| P3 | K-Means, k = 3 | Silhouette 0.09 · Davies-Bouldin 2.61 |

Los resultados P2 de esa tabla son una referencia histórica: la implementación actual
purga las tres horas previas al test para impedir fuga en el target de horizonte de tres
horas. Ejecuta de nuevo el baseline y los candidatos para compararlos bajo el mismo corte.

### Decisiones de diseño

- **Target de P2:** los datos abiertos no incluyen la ocupación real de las estaciones. Por eso se aproxima como `|balance neto acumulado en t+1..t+3| ≥ 30 % de dpcapacity`. Los dos valores son constantes en `src/train_clasificacion.py`.
- **P3:** el perfil de cada estación se calcula solo con el periodo de entrenamiento. La latitud y la longitud se usan para el mapa, no para agrupar.
- **Calidad de datos:** se eliminan los `Viaje_Id` duplicados, los viajes de menos de 1 minuto o más de 24 horas y las estaciones fuera del AMG o sin registro en la nomenclatura.
- **Privacidad:** `Usuario_Id`, `Genero` y `Anio_de_nacimiento` no se usan como variables de los modelos.

## Parte C — Model Registry y reporte

Los modelos seleccionados se registran en el Model Registry con nombres estables y se
promueven con el alias `champion`. Se usan aliases en vez de las etapas legacy (`Staging`,
`Production`), porque son el esquema vigente de MLflow y permiten cambiar la versión de
producción sin recrear el modelo registrado.

| Pregunta | Nombre en Model Registry | URI de producción |
|---|---|---|
| P1 — demanda | `mibici-equipo33-p1-demanda` | `models:/mibici-equipo33-p1-demanda@champion` |
| P2 — riesgo | `mibici-equipo33-p2-riesgo` | `models:/mibici-equipo33-p2-riesgo@champion` |
| P3 — grupos | `mibici-equipo33-p3-grupos` | `models:/mibici-equipo33-p3-grupos@champion` |

Después de comparar los runs baseline y candidatos de cada pregunta en la UI de MLflow,
registra los tres IDs elegidos. El backend debe estar activo, `MLFLOW_TRACKING_URI` debe
apuntar a él y los runs seleccionados deben provenir de un árbol Git limpio:

```powershell
python -m src.register_models `
  --p1-run-id <RUN_ID_P1> --p1-justification "Justificación basada en RMSE, MAE, R2 y costo." `
  --p2-run-id <RUN_ID_P2> --p2-justification "Justificación basada en recall, PR-AUC, F1 y costo." `
  --p3-run-id <RUN_ID_P3> --p3-justification "Justificación basada en silhouette, Davies-Bouldin y costo."
```

El comando valida experimento, pregunta, etapa, métricas, commit, limpieza del árbol,
versión de datos, firma y carga del modelo. Registra cada versión desde su run de origen,
agrega tags con el experimento, `run_id`, commit y `data_version`, asigna el alias
`champion` y verifica la carga por URI de producción. Una promoción nueva mueve el alias
al run que se indique. El reporte [reports/parte-c-model-registry.md](reports/parte-c-model-registry.md)
documenta los criterios y, después de una promoción exitosa, contiene las métricas reales,
duraciones y justificaciones de los runs seleccionados.

La selección debe privilegiar `test_rmse` bajo para P1, `test_recall` y `test_pr_auc` para
P2 por el costo de no detectar estaciones en riesgo, y `silhouette` junto con
`davies_bouldin` para P3. La duración del entrenamiento se reporta como referencia de
costo; el costo de inferencia no está medido.

Para cargar un modelo promovido:

```powershell
python -c "import mlflow.pyfunc; mlflow.pyfunc.load_model('models:/mibici-equipo33-p1-demanda@champion')"
```
