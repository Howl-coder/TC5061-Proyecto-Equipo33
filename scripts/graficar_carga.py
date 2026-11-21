"""Grafica el historial agregado CSV generado por Locust."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def leer_historial(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    if not filas:
        raise ValueError(f"No hay estadísticas agregadas en {ruta}")
    if any(fila["Name"] != "Aggregated" for fila in filas):
        raise ValueError("Usa el historial generado sin --csv-full-history")

    inicio = int(filas[0]["Timestamp"])
    historial = []
    for fila in filas:
        solicitudes = int(fila["Total Request Count"])
        fallos = int(fila["Total Failure Count"])
        if solicitudes < 0 or fallos < 0 or fallos > solicitudes:
            raise ValueError("Los conteos de solicitudes y fallos son inválidos")
        historial.append(
            {
                "segundos": int(fila["Timestamp"]) - inicio,
                "p50": float(fila["50%"]) if solicitudes else math.nan,
                "p95": float(fila["95%"]) if solicitudes else math.nan,
                "rps": float(fila["Requests/s"]),
                "errores": 100 * fallos / solicitudes if solicitudes else math.nan,
            }
        )
    return historial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historial", type=Path, help="Archivo *_stats_history.csv de Locust")
    parser.add_argument("salida", type=Path, help="Ruta de la imagen PNG")
    args = parser.parse_args()
    datos = leer_historial(args.historial)
    tiempo = [dato["segundos"] for dato in datos]

    fig, ejes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    ejes[0].plot(tiempo, [dato["p50"] for dato in datos], label="p50")
    ejes[0].plot(tiempo, [dato["p95"] for dato in datos], label="p95")
    ejes[0].set_ylabel("Latencia (ms)")
    ejes[0].set_title("Latencia acumulada")
    ejes[0].legend()
    ejes[1].plot(tiempo, [dato["rps"] for dato in datos], color="tab:green")
    ejes[1].set_ylabel("Peticiones/s")
    ejes[1].set_title("Peticiones por segundo (ventana de Locust)")
    ejes[2].plot(tiempo, [dato["errores"] for dato in datos], color="tab:red")
    ejes[2].set_ylabel("Errores (%)")
    ejes[2].set_ylim(-2, 102)
    ejes[2].set_yticks([0, 25, 50, 75, 100])
    ejes[2].set_title("Tasa de error acumulada")
    ejes[2].set_xlabel("Segundos desde la primera muestra")
    for eje in ejes:
        eje.grid(alpha=0.25)
    fig.savefig(args.salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
