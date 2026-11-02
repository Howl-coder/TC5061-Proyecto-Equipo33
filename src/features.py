"""Ingeniería de variables para las tres preguntas del proyecto MiBici.

Entrada: ``data/interim/viajes_amg_limpios.parquet`` (generado por ``src.dataset``).
Salida:  ``data/processed/features_demanda_p1.parquet`` (malla hora x estación con variables).

Uso:
    python -m src.features

Diccionario de variables
------------------------
Malla horaria (una fila por estación ``Origen_Id`` y hora ``timestamp_hora``):

- ``demanda_salidas``: viajes que inician en la estación durante la hora. **Target de P1.**
- ``flujo_llegadas``: viajes que terminan en la estación durante la hora.
- ``balance_neto_h``: ``flujo_llegadas - demanda_salidas``. Positivo = acumula bicis.

Calendario (codificación cíclica para que 23 h y 0 h, o domingo y lunes, queden cerca):

- ``hora_sin``, ``hora_cos``: hora del día en el círculo unitario (periodo 24).
- ``dia_sin``, ``dia_cos``: día de la semana en el círculo unitario (periodo 7).
- ``es_fin_de_semana``: 1 si sábado o domingo.

Historia de la propia estación (solo usan horas anteriores a ``t``):

- ``lag_salidas_{1,2,3}h``: inercia de corto plazo.
- ``lag_salidas_24h``: misma hora del día anterior (estacionalidad diaria).
- ``lag_salidas_168h``: misma hora de la semana anterior (estacionalidad semanal).
- ``rolling_mean_6h``: media de salidas en ``t-6 .. t-1``.

P2 (agregadas por ``agregar_features_y_target_desbalance``):

- ``dpcapacity``: anclajes instalados en la estación.
- ``lag_balance_{1,2}h``: balance neto de las horas ``t-1`` y ``t-2``.
- ``balance_acum_3h``: suma del balance neto en ``t-2 .. t``.
- ``en_riesgo_desbalance``: **target de P2** (ver la función).

P3 (vector por estación, ``construir_perfil_estaciones``):

- ``prop_hora_00 .. prop_hora_23``: proporción de salidas en cada hora del día.
- ``prop_fin_de_semana``, ``prop_circulares``, ``prop_salidas``, ``viajes_por_dia``,
  ``uso_por_capacidad``: intensidad y tipo de uso de la estación.
"""
from config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
import numpy as np
import pandas as pd


def generar_malla_demanda_horaria(df_viajes: pd.DataFrame, ids_estaciones: list) -> pd.DataFrame:
    """Construye la malla regular hora x estación con imputación de demanda cero.

    Una hora sin viajes no aparece en los datos crudos; si no se rellena, el modelo nunca
    vería los ceros (madrugadas, estaciones periféricas) y sobreestimaría la demanda.

    Args:
        df_viajes: viajes limpios con ``Inicio_del_viaje``, ``Fin_del_viaje``,
            ``Origen_Id`` y ``Destino_Id``.
        ids_estaciones: estaciones que forman la malla.

    Returns:
        DataFrame con una fila por (``Origen_Id``, ``timestamp_hora``) y las columnas
        ``demanda_salidas`` y ``flujo_llegadas`` (0 cuando no hubo viajes),
        ordenado por estación y hora.
    """
    # Truncar inicio a la hora
    df_viajes["timestamp_hora"] = df_viajes["Inicio_del_viaje"].dt.floor("h")

    # Demanda de salidas por estación y hora
    demanda = (
        df_viajes.groupby(["timestamp_hora", "Origen_Id"])
        .size()
        .reset_index(name="demanda_salidas")
    )

    # Flujo de llegadas por estación y hora (para P2)
    df_viajes["timestamp_fin_hora"] = df_viajes["Fin_del_viaje"].dt.floor("h")
    llegadas = (
        df_viajes.groupby(["timestamp_fin_hora", "Destino_Id"])
        .size()
        .reset_index(name="flujo_llegadas")
        .rename(columns={"timestamp_fin_hora": "timestamp_hora", "Destino_Id": "Origen_Id"})
    )

    # Rango horario completo continuo
    rango_horas = pd.date_range(
        start=df_viajes["timestamp_hora"].min(),
        end=max(
            df_viajes["timestamp_hora"].max(),
            df_viajes["timestamp_fin_hora"].max(),
        ),
        freq="h",
    )

    # Producto cartesiano completo
    indice_cartesiano = pd.MultiIndex.from_product(
        [rango_horas, ids_estaciones], names=["timestamp_hora", "Origen_Id"]
    ).to_frame().reset_index(drop=True)

    # Combinar y rellenar ceros
    malla = indice_cartesiano.merge(demanda, on=["timestamp_hora", "Origen_Id"], how="left")
    malla = malla.merge(llegadas, on=["timestamp_hora", "Origen_Id"], how="left")

    malla["demanda_salidas"] = malla["demanda_salidas"].fillna(0).astype(int)
    malla["flujo_llegadas"] = malla["flujo_llegadas"].fillna(0).astype(int)

    malla.sort_values(by=["Origen_Id", "timestamp_hora"], inplace=True)
    malla.reset_index(drop=True, inplace=True)
    return malla

def calcular_features_temporales_y_lags(df_malla: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables cíclicas de calendario, rezagos y media móvil por estación.

    Todos los rezagos y la media móvil se calculan dentro de cada estación y solo con
    horas anteriores a ``t`` (``shift`` >= 1), así que no hay fuga del futuro.

    Args:
        df_malla: salida de :func:`generar_malla_demanda_horaria`.

    Returns:
        Copia de la malla con las variables del diccionario del módulo. Se descartan las
        primeras 168 horas de cada estación porque aún no tienen ``lag_salidas_168h``.
    """
    df = df_malla.copy()

    # Features cíclicas
    hora = df["timestamp_hora"].dt.hour
    dia_semana = df["timestamp_hora"].dt.dayofweek

    df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    df["dia_sin"] = np.sin(2 * np.pi * dia_semana / 7)
    df["dia_cos"] = np.cos(2 * np.pi * dia_semana / 7)
    df["es_fin_de_semana"] = (dia_semana >= 5).astype(int)

    # Rezagos temporales por estación (1h, 2h, 3h, 24h, 168h)
    for lag in [1, 2, 3, 24, 168]:
        df[f"lag_salidas_{lag}h"] = df.groupby("Origen_Id")["demanda_salidas"].shift(lag)

    # Media móvil de las últimas 6 horas (shift(1) para evitar data leakage)
    df["rolling_mean_6h"] = df.groupby("Origen_Id")["demanda_salidas"].transform(
        lambda x: x.shift(1).rolling(6).mean()
    )

    # Balance instantáneo para P2
    df["balance_neto_h"] = df["flujo_llegadas"] - df["demanda_salidas"]

    # Descartar filas con NaN derivados de los lags iniciales
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def agregar_features_y_target_desbalance(
    df_features: pd.DataFrame,
    df_estaciones: pd.DataFrame,
    horizonte_h: int,
    umbral_fraccion_capacidad: float,
) -> pd.DataFrame:
    """P2: agrega capacidad, rezagos de balance y el target ``en_riesgo_desbalance``.

    El target vale 1 si el balance neto acumulado en las horas ``t+1 .. t+horizonte_h``
    alcanza, en valor absoluto, ``umbral_fraccion_capacidad * dpcapacity``. Un balance
    muy negativo indica riesgo de quedarse sin bicis y uno muy positivo, sin anclajes.
    Es una aproximación: los datos abiertos no incluyen la ocupación real de la estación.

    Args:
        df_features: salida de :func:`calcular_features_temporales_y_lags`.
        df_estaciones: catálogo con ``id`` y ``dpcapacity``.
        horizonte_h: horas hacia adelante que cubre el target.
        umbral_fraccion_capacidad: fracción de la capacidad que se considera crítica.

    Returns:
        DataFrame con las variables de P2 y el target. Se eliminan las últimas
        ``horizonte_h`` horas de cada estación porque su futuro no está completo.
    """
    capacidad = df_estaciones[["id", "dpcapacity"]].rename(columns={"id": "Origen_Id"})
    df = df_features.merge(capacidad, on="Origen_Id", how="inner")
    df.sort_values(by=["Origen_Id", "timestamp_hora"], inplace=True)

    grupo = df.groupby("Origen_Id")["balance_neto_h"]
    df["lag_balance_1h"] = grupo.shift(1)
    df["lag_balance_2h"] = grupo.shift(2)
    df["balance_acum_3h"] = grupo.transform(lambda x: x.rolling(3).sum())

    balance_futuro = sum(grupo.shift(-k) for k in range(1, horizonte_h + 1))
    df["en_riesgo_desbalance"] = (
        balance_futuro.abs() >= umbral_fraccion_capacidad * df["dpcapacity"]
    ).astype(int)

    # Las últimas horas no tienen futuro observado completo
    df = df[balance_futuro.notna()].dropna()
    df.reset_index(drop=True, inplace=True)
    return df

def construir_perfil_estaciones(
    df_viajes: pd.DataFrame, df_estaciones: pd.DataFrame, fecha_fin: str
) -> pd.DataFrame:
    """P3: construye el vector de comportamiento de cada estación para clustering.

    Solo usa viajes hasta ``fecha_fin`` (fin del periodo de entrenamiento), para que los
    grupos no dependan del mes de prueba.

    Args:
        df_viajes: viajes limpios.
        df_estaciones: catálogo con ``id``, ``name``, ``latitude``, ``longitude`` y
            ``dpcapacity``.
        fecha_fin: última fecha incluida, por ejemplo ``config.FECHA_FIN_TRAIN``.

    Returns:
        Una fila por estación con las variables P3 del diccionario del módulo, más
        nombre, coordenadas y capacidad (estas últimas solo para graficar el mapa).
    """
    df = df_viajes[df_viajes["Inicio_del_viaje"] <= fecha_fin]
    n_dias = df["Inicio_del_viaje"].dt.normalize().nunique()

    hora = df["Inicio_del_viaje"].dt.hour
    perfil_horario = pd.crosstab(df["Origen_Id"], hora, normalize="index")
    perfil_horario = perfil_horario.reindex(columns=range(24), fill_value=0)
    perfil_horario.columns = [f"prop_hora_{h:02d}" for h in range(24)]

    salidas = df.groupby("Origen_Id").size().rename("salidas")
    llegadas = df.groupby("Destino_Id").size().rename("llegadas")
    fin_semana = (
        df["Inicio_del_viaje"].dt.dayofweek.ge(5).groupby(df["Origen_Id"]).mean()
        .rename("prop_fin_de_semana")
    )
    circulares = (
        (df["Origen_Id"] == df["Destino_Id"]).groupby(df["Origen_Id"]).mean()
        .rename("prop_circulares")
    )

    perfil = pd.concat([perfil_horario, salidas, llegadas, fin_semana, circulares], axis=1)
    perfil = perfil.dropna(subset=["salidas"]).fillna(0)
    perfil.index.name = "id"

    perfil = perfil.join(
        df_estaciones.set_index("id")[["name", "latitude", "longitude", "dpcapacity"]],
        how="inner",
    )
    perfil["prop_salidas"] = perfil["salidas"] / (perfil["salidas"] + perfil["llegadas"])
    perfil["viajes_por_dia"] = perfil["salidas"] / n_dias
    perfil["uso_por_capacidad"] = perfil["viajes_por_dia"] / perfil["dpcapacity"].clip(lower=1)
    return perfil.reset_index()

if __name__ == "__main__":
    ruta_interim = INTERIM_DATA_DIR / "viajes_amg_limpios.parquet"
    if not ruta_interim.exists():
        raise FileNotFoundError(f"Primero corre dataset.py para generar {ruta_interim}")

    print(f"Cargando {ruta_interim.name}...")
    df_viajes = pd.read_parquet(ruta_interim)

    estaciones = sorted(df_viajes["Origen_Id"].unique())
    print(f"Generando malla horaria para {len(estaciones)} estaciones...")
    df_malla = generar_malla_demanda_horaria(df_viajes, estaciones)

    print("Calculando features temporales, cíclicas y lags...")
    df_features = calcular_features_temporales_y_lags(df_malla)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    salida = PROCESSED_DATA_DIR / "features_demanda_p1.parquet"
    df_features.to_parquet(salida, index=False)

    print(f"Features P1 guardadas con éxito en: {salida}")
    print(f"Dimensiones de entrenamiento: {df_features.shape}")