"""Comprueba el linaje de la exportación P1 antes de publicar una imagen."""

import json
from pathlib import Path
import re
import sys


def version_publicable(ruta: Path) -> str:
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if not isinstance(datos, dict) or not isinstance(datos.get("tags"), dict):
        raise TypeError("El modelo champion no cumple los requisitos de linaje.")
    tags = datos.get("tags", {})
    version = datos.get("version")
    commit = tags.get("git_commit", "")
    if (
        datos.get("nombre") != "mibici-equipo33-p1-demanda"
        or datos.get("alias") != "champion"
        or not isinstance(version, str)
        or not version.isascii()
        or not version.isdigit()
        or int(version) < 1
        or not datos.get("run_id")
        or tags.get("source_run_id") != datos["run_id"]
        or tags.get("question") != "P1"
        or tags.get("git_dirty") != "False"
        or not isinstance(commit, str)
        or not re.fullmatch(r"[0-9a-f]{40,64}", commit)
        or not tags.get("data_version")
    ):
        raise ValueError("El modelo champion no cumple los requisitos de linaje.")
    return version


if __name__ == "__main__":
    try:
        print(version_publicable(Path("models/servicio/registro.json")))
    except (OSError, ValueError, TypeError) as error:
        sys.exit(f"No se puede publicar: {error}")
