"""Descarga del Model Registry el modelo que se empaqueta en la imagen del servicio.

Se ejecuta en la máquina que tiene acceso al servidor de MLflow (MLFLOW_TRACKING_URI);
el contenedor solo lee la copia local, sin depender del servidor.

Ejemplos:
    python -m src.model_package --pregunta p1                  # alias champion
    python -m src.model_package --pregunta p1 --version 3
    python -m src.model_package --nombre mibici-equipo33-p1-demanda --alias champion
"""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import tempfile

from config import (
    MLFLOW_PRODUCTION_ALIAS,
    MLFLOW_REGISTERED_MODELS,
    MLFLOW_TRACKING_URI,
    SERVING_MODEL_DIR,
)
import mlflow
from mlflow import MlflowClient
import mlflow.artifacts
import mlflow.pyfunc

ARCHIVO_METADATOS = "registro.json"


def uri_registro(nombre: str, version: str | int | None = None, alias: str | None = None) -> str:
    """URI ``models:/`` por versión (``models:/<nombre>/<n>``) o por alias (``@<alias>``)."""
    if (version is None) == (alias is None):
        raise ValueError("Indica exactamente uno: versión o alias.")
    if version is not None:
        if not str(version).isdigit():
            raise ValueError(f"Versión inválida: {version!r}; debe ser un entero positivo.")
        return f"models:/{nombre}/{version}"
    return f"models:/{nombre}@{alias}"


def exportar_modelo(
    nombre: str,
    destino: Path = SERVING_MODEL_DIR,
    version: str | int | None = None,
    alias: str | None = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> dict:
    """Descarga una versión registrada a ``destino`` y guarda su linaje en registro.json."""
    uri_solicitada = uri_registro(nombre, version, alias)
    destino = Path(destino)
    if destino.exists() and not (destino / ARCHIVO_METADATOS).is_file():
        raise FileExistsError(
            f"{destino} existe y no es una exportación previa; no se sobrescribe."
        )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    if alias is not None:
        model_version = client.get_model_version_by_alias(nombre, alias)
    else:
        model_version = client.get_model_version(nombre, str(version))
    version_registrada = str(model_version.version)
    # Descargar por número de versión evita que un cambio de alias mezcle versiones
    uri_version = uri_registro(nombre, version=version_registrada)

    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri_version, dst_path=tmp))
        modelo = mlflow.pyfunc.load_model(str(ruta))
        if modelo.metadata.signature is None:
            raise ValueError(f"El modelo {uri_version} no tiene firma MLflow.")

        metadatos = {
            "nombre": nombre,
            "version": version_registrada,
            "alias": alias,
            "uri_solicitada": uri_solicitada,
            "uri_version": uri_version,
            "run_id": model_version.run_id,
            "tags": dict(model_version.tags),
            "exportado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (ruta / ARCHIVO_METADATOS).write_text(
            json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if destino.exists():
            shutil.rmtree(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ruta, destino)
    return metadatos


def cargar_modelo_empaquetado(ruta: Path = SERVING_MODEL_DIR):
    """Carga el modelo exportado y sus metadatos de registro; usado dentro del contenedor."""
    ruta = Path(ruta)
    ruta_metadatos = ruta / ARCHIVO_METADATOS
    if not ruta_metadatos.is_file():
        raise FileNotFoundError(
            f"No hay un modelo exportado en {ruta}; ejecuta `python -m src.model_package`."
        )
    metadatos = json.loads(ruta_metadatos.read_text(encoding="utf-8"))
    modelo = mlflow.pyfunc.load_model(str(ruta))
    if modelo.metadata.signature is None:
        raise ValueError(f"El modelo en {ruta} no tiene firma MLflow.")
    return modelo, metadatos


def _argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modelo = parser.add_mutually_exclusive_group()
    modelo.add_argument("--pregunta", choices=sorted(MLFLOW_REGISTERED_MODELS), default="p1")
    modelo.add_argument("--nombre", help="Nombre del modelo en el Model Registry.")
    referencia = parser.add_mutually_exclusive_group()
    referencia.add_argument("--version", help="Número de versión registrada.")
    referencia.add_argument("--alias", help=f"Alias (por defecto: {MLFLOW_PRODUCTION_ALIAS}).")
    parser.add_argument("--destino", type=Path, default=SERVING_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = _argparse()
    nombre = args.nombre or MLFLOW_REGISTERED_MODELS[args.pregunta]
    alias = None if args.version else (args.alias or MLFLOW_PRODUCTION_ALIAS)

    metadatos = exportar_modelo(nombre, args.destino, version=args.version, alias=alias)
    print(
        f"Exportado {metadatos['uri_solicitada']} -> versión {metadatos['version']} "
        f"(run {metadatos['run_id']}) en {args.destino}"
    )

    modelo, _ = cargar_modelo_empaquetado(args.destino)
    ejemplo = modelo.metadata.load_input_example(str(args.destino))
    if ejemplo is not None:
        print(f"Predicción de verificación con input_example: {modelo.predict(ejemplo)}")


if __name__ == "__main__":
    main()
