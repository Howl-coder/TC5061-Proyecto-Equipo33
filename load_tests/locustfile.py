"""Carga de predicciones P1 con una estación por petición."""

import math
import random

from locust import HttpUser, between, task


def instancia() -> dict:
    hora = random.randrange(24)
    dia = random.randrange(7)
    return {
        "hora_sin": math.sin(2 * math.pi * hora / 24),
        "hora_cos": math.cos(2 * math.pi * hora / 24),
        "dia_sin": math.sin(2 * math.pi * dia / 7),
        "dia_cos": math.cos(2 * math.pi * dia / 7),
        "es_fin_de_semana": int(dia >= 5),
        "lag_salidas_1h": random.randint(0, 10),
        "lag_salidas_2h": random.randint(0, 10),
        "lag_salidas_3h": random.randint(0, 10),
        "lag_salidas_24h": random.randint(0, 10),
        "lag_salidas_168h": random.randint(0, 10),
        "rolling_mean_6h": random.uniform(0, 10),
    }


class UsuarioPrediccion(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def predecir(self):
        with self.client.post(
            "/predict",
            json={"instancias": [instancia()]},
            timeout=30,
            catch_response=True,
        ) as respuesta:
            if respuesta.status_code != 200:
                if respuesta.status_code:
                    respuesta.failure(f"HTTP {respuesta.status_code}")
                return
            try:
                cuerpo = respuesta.json()
            except ValueError:
                respuesta.failure("La respuesta no es JSON válido")
                return

            if not isinstance(cuerpo, dict):
                respuesta.failure("La respuesta no es un objeto JSON")
                return
            predicciones = cuerpo.get("predicciones")
            if (
                cuerpo.get("target") != "demanda_salidas"
                or not isinstance(cuerpo.get("modelo"), str)
                or not isinstance(cuerpo.get("version"), str)
                or not isinstance(predicciones, list)
                or len(predicciones) != 1
                or isinstance(predicciones[0], bool)
                or not isinstance(predicciones[0], (int, float))
                or not math.isfinite(predicciones[0])
                or predicciones[0] < 0
            ):
                respuesta.failure("La respuesta no cumple el contrato de predicción")
