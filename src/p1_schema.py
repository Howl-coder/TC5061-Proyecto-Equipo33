"""Target y variables de entrada compartidos por entrenamiento y validación P1."""

TARGET = "demanda_salidas"
FEATURES = [
    "hora_sin",
    "hora_cos",
    "dia_sin",
    "dia_cos",
    "es_fin_de_semana",
    "lag_salidas_1h",
    "lag_salidas_2h",
    "lag_salidas_3h",
    "lag_salidas_24h",
    "lag_salidas_168h",
    "rolling_mean_6h",
]
