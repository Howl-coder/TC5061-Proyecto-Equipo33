"""API REST del modelo P1: salidas esperadas de una estación MiBici en la próxima hora.

Carga el modelo exportado con ``python -m src.model_package`` (carpeta MODELO_SERVICIO_DIR).

Uso local:
    uvicorn src.api:app --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager
import math
from pathlib import Path
from typing import Annotated, Literal

from config import SERVING_MODEL_DIR
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.model_package import cargar_modelo_empaquetado
from src.p1_schema import FEATURES, TARGET

MAX_INSTANCIAS = 1000
# Holguras para clientes que redondean senos/cosenos (p. ej. a 3 decimales)
TOLERANCIA_CIRCULO = 0.01
TOLERANCIA_POSICION = 0.1


def _posicion_ciclica(seno: float, coseno: float, periodo: int) -> float:
    return (math.atan2(seno, coseno) * periodo / (2 * math.pi)) % periodo


class EntradaDemanda(BaseModel):
    """Variables de una estación en la hora a pronosticar, como las genera src.features."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "hora_sin": 0.866,
                    "hora_cos": -0.5,
                    "dia_sin": 0.0,
                    "dia_cos": 1.0,
                    "es_fin_de_semana": 0,
                    "lag_salidas_1h": 3,
                    "lag_salidas_2h": 2,
                    "lag_salidas_3h": 4,
                    "lag_salidas_24h": 5,
                    "lag_salidas_168h": 4,
                    "rolling_mean_6h": 3.2,
                }
            ]
        },
    )

    hora_sin: float = Field(ge=-1, le=1, description="sin(2π·hora/24)")
    hora_cos: float = Field(ge=-1, le=1, description="cos(2π·hora/24)")
    dia_sin: float = Field(ge=-1, le=1, description="sin(2π·día_semana/7), lunes = 0")
    dia_cos: float = Field(ge=-1, le=1, description="cos(2π·día_semana/7), lunes = 0")
    es_fin_de_semana: int = Field(ge=0, le=1, description="1 si es sábado o domingo")
    lag_salidas_1h: float = Field(ge=0, description="Salidas de la hora anterior")
    lag_salidas_2h: float = Field(ge=0, description="Salidas de hace 2 horas")
    lag_salidas_3h: float = Field(ge=0, description="Salidas de hace 3 horas")
    lag_salidas_24h: float = Field(ge=0, description="Salidas a la misma hora del día anterior")
    lag_salidas_168h: float = Field(ge=0, description="Salidas a la misma hora, semana anterior")
    rolling_mean_6h: float = Field(ge=0, description="Media de salidas de t-6 a t-1")

    @model_validator(mode="after")
    def _validar_calendario(self) -> "EntradaDemanda":
        for prefijo, periodo in (("hora", 24), ("dia", 7)):
            seno = getattr(self, f"{prefijo}_sin")
            coseno = getattr(self, f"{prefijo}_cos")
            if abs(seno**2 + coseno**2 - 1) > TOLERANCIA_CIRCULO:
                raise ValueError(f"{prefijo}_sin y {prefijo}_cos deben cumplir sin² + cos² = 1.")
            posicion = _posicion_ciclica(seno, coseno, periodo)
            if abs(posicion - round(posicion)) > TOLERANCIA_POSICION:
                raise ValueError(
                    f"{prefijo}_sin y {prefijo}_cos no codifican un valor entero "
                    f"de {prefijo} (periodo {periodo})."
                )
        dia_semana = round(_posicion_ciclica(self.dia_sin, self.dia_cos, 7)) % 7
        if self.es_fin_de_semana != int(dia_semana >= 5):
            raise ValueError("es_fin_de_semana no coincide con el día de dia_sin/dia_cos.")
        return self


class SolicitudPrediccion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instancias: list[EntradaDemanda] = Field(
        min_length=1, max_length=MAX_INSTANCIAS, description="Estaciones-hora a pronosticar"
    )


class RespuestaPrediccion(BaseModel):
    modelo: str = Field(description="Nombre en el Model Registry")
    version: str = Field(description="Versión registrada que se sirve")
    target: str = Field(description="Variable pronosticada")
    predicciones: list[Annotated[float, Field(ge=0)]] = Field(
        description="Salidas esperadas en la próxima hora, en el orden de `instancias`"
    )


class RespuestaModelo(BaseModel):
    nombre: str
    version: str
    alias: str | None
    run_id: str | None
    target: str
    features: list[str]


class RespuestaSalud(BaseModel):
    status: Literal["ok"]


def _validar_features(modelo) -> None:
    esperadas = modelo.metadata.get_input_schema().input_names()
    if list(esperadas) != FEATURES:
        raise ValueError(
            f"El modelo empaquetado espera {esperadas}; la API sirve P1 con {FEATURES}."
        )


def crear_app(ruta_modelo: Path = SERVING_MODEL_DIR) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        modelo, metadatos = cargar_modelo_empaquetado(ruta_modelo)
        _validar_features(modelo)
        app.state.modelo = modelo
        app.state.metadatos = metadatos
        yield

    app = FastAPI(
        title="MiBici — demanda por estación (P1)",
        description="Pronóstico de salidas de bicicletas por estación en la próxima hora.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def error_validacion(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Sin "input" (no se devuelve el payload; NaN/Infinity no son JSON) ni "ctx" (excepciones)
        errores = [
            {k: v for k, v in error.items() if k not in {"input", "ctx"}} for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errores)})

    @app.get("/health")
    def health() -> RespuestaSalud:
        return RespuestaSalud(status="ok")

    @app.get("/modelo")
    def info_modelo(request: Request) -> RespuestaModelo:
        metadatos = request.app.state.metadatos
        return RespuestaModelo(
            nombre=metadatos["nombre"],
            version=str(metadatos["version"]),
            alias=metadatos.get("alias"),
            run_id=metadatos.get("run_id"),
            target=TARGET,
            features=FEATURES,
        )

    @app.post("/predict")
    def predict(solicitud: SolicitudPrediccion, request: Request) -> RespuestaPrediccion:
        entrada = pd.DataFrame(
            [instancia.model_dump() for instancia in solicitud.instancias], columns=FEATURES
        )
        # Igual que en la evaluación: una demanda negativa no tiene sentido
        predicciones = np.clip(np.asarray(request.app.state.modelo.predict(entrada)), 0, None)
        metadatos = request.app.state.metadatos
        return RespuestaPrediccion(
            modelo=metadatos["nombre"],
            version=str(metadatos["version"]),
            target=TARGET,
            predicciones=predicciones.ravel().tolist(),
        )

    return app


app = crear_app()
