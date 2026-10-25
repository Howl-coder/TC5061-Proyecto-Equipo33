"""Ingesta, versionado y limpieza de los viajes MiBici.

Uso:
    python -m src.dataset                          # descarga si faltan, verifica y procesa
    python -m src.dataset --actualizar-manifiesto  # registra una nueva versión de los datos crudos
    python -m src.dataset --forzar-descarga        # re-descarga aunque los archivos ya existan
"""
import argparse
import gc
import hashlib
import json
import urllib.request
from pathlib import Path
import pandas as pd
from config import (
    ARCHIVOS_VIAJES_RAW,
    RUTA_CATALOGO_ESTACIONES,
    RUTA_MANIFIESTO_DATOS,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    DATA_VERSION,
    FECHA_INICIO,
    FECHA_FIN,
    DURACION_MIN_MINUTOS,
    DURACION_MAX_MINUTOS,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    URLS_VIAJES_RAW,
    URL_CATALOGO_ESTACIONES,
)

ARCHIVOS_CRUDOS = [*ARCHIVOS_VIAJES_RAW, RUTA_CATALOGO_ESTACIONES]
# El portal responde 406 a menos que lleguen ambos encabezados
HEADERS_DESCARGA = {"Accept": "*/*", "User-Agent": "mibici-equipo33"}


def descargar_datos_crudos(forzar: bool = False) -> None:
    """Descarga los CSV mensuales de viajes y la nomenclatura de estaciones desde el portal MiBici.

    Usa streaming por bloques para no cargar cada archivo en RAM. Si el archivo
    ya existe y forzar=False, se omite para no repetir descargas innecesarias.

    Args:
        forzar: si True, re-descarga aunque los archivos ya existan localmente.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    descargas = {
        **URLS_VIAJES_RAW,
        RUTA_CATALOGO_ESTACIONES.name: URL_CATALOGO_ESTACIONES,
    }

    for nombre, url in descargas.items():
        destino = RAW_DATA_DIR / nombre
        if destino.exists() and not forzar:
            print(f"Ya existe, omitiendo: {nombre}")
            continue
        print(f"Descargando {nombre} ...")
        solicitud = urllib.request.Request(url, headers=HEADERS_DESCARGA)
        with urllib.request.urlopen(solicitud) as respuesta, open(destino, "wb") as archivo:
            while bloque := respuesta.read(1 << 20):
                archivo.write(bloque)
        print(f"  -> {destino.stat().st_size / 1e6:.1f} MB")


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def generar_manifiesto(archivos: list) -> dict:
    """Calcula tamaño y SHA-256 de cada archivo crudo y un hash global de la versión."""
    faltantes = [str(p) for p in archivos if not p.exists()]
    if faltantes:
        raise FileNotFoundError(f"Faltan archivos crudos en data/raw/: {faltantes}")

    entradas = [
        {"archivo": p.name, "bytes": p.stat().st_size, "sha256": sha256_archivo(p)}
        for p in archivos
    ]
    huella = "\n".join(f"{e['archivo']}  {e['sha256']}" for e in entradas)
    return {
        "version": DATA_VERSION,
        "sha256_global": hashlib.sha256(huella.encode()).hexdigest(),
        "archivos": entradas,
    }


def cargar_manifiesto() -> dict:
    with open(RUTA_MANIFIESTO_DATOS, encoding="utf-8") as f:
        return json.load(f)


def guardar_manifiesto(manifiesto: dict) -> None:
    RUTA_MANIFIESTO_DATOS.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_MANIFIESTO_DATOS, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)
        f.write("\n")


def verificar_datos_crudos(actualizar: bool = False) -> dict:
    """Compara los CSV locales contra el manifiesto versionado y devuelve el manifiesto vigente.

    Si no existe manifiesto, o si actualizar=True, se registra la versión actual de los datos.
    Si los archivos difieren del manifiesto, se detiene para no mezclar versiones sin querer.
    """
    actual = generar_manifiesto(ARCHIVOS_CRUDOS)
    if actualizar or not RUTA_MANIFIESTO_DATOS.exists():
        guardar_manifiesto(actual)
        print(f"Manifiesto registrado: {RUTA_MANIFIESTO_DATOS} ({actual['sha256_global'][:16]})")
        return actual

    registrado = cargar_manifiesto()
    if registrado["sha256_global"] != actual["sha256_global"]:
        esperados = {e["archivo"]: e["sha256"] for e in registrado["archivos"]}
        distintos = [e["archivo"] for e in actual["archivos"] if esperados.get(e["archivo"]) != e["sha256"]]
        raise ValueError(
            f"Los datos crudos no coinciden con {RUTA_MANIFIESTO_DATOS.name} "
            f"(archivos distintos: {distintos}). Si es una nueva versión intencional, "
            "ejecuta: python -m src.dataset --actualizar-manifiesto"
        )
    print(f"Datos crudos verificados: {registrado['version']} ({registrado['sha256_global'][:16]})")
    return registrado


def cargar_catalogo_amg(ruta_estaciones: Path) -> pd.DataFrame:
    """Carga y filtra estaciones estrictamente dentro del polígono AMG."""
    try:
        df_est = pd.read_csv(ruta_estaciones, encoding="utf-8")
    except UnicodeDecodeError:
        df_est = pd.read_csv(ruta_estaciones, encoding="latin-1")
    
    # Filtrado por coordenadas AMG
    mask_amg = (
        (df_est["latitude"] >= LAT_MIN) & (df_est["latitude"] <= LAT_MAX) &
        (df_est["longitude"] >= LON_MIN) & (df_est["longitude"] <= LON_MAX)
    )
    df_est = df_est[mask_amg].copy()
    df_est["id"] = df_est["id"].astype(int)
    return df_est

def procesar_e_integrar_datos(archivos: list, ruta_estaciones: Path) -> pd.DataFrame:
    """Lee archivos explícitos, limpia nombres, optimiza memoria y filtra al AMG."""
    df_est = cargar_catalogo_amg(ruta_estaciones)
    estaciones_validas = set(df_est["id"].unique())
    print(f"Estaciones válidas en AMG: {len(estaciones_validas)}")

    dfs = []
    for f in archivos:
        print(f"Leyendo: {f.name}...")
        try:
            temp = pd.read_csv(f, encoding="utf-8")
        except UnicodeDecodeError:
            temp = pd.read_csv(f, encoding="latin-1")

        # Unificar nombres de columnas
        temp.columns = (
            temp.columns.str.replace("Año", "Anio")
            .str.replace("AÃ±o", "Anio")
            .str.strip()
        )

        if "Genero" in temp.columns:
            temp["Genero"] = temp["Genero"].astype("category")

        dfs.append(temp)

    # Concatenar y liberar memoria
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    print(f"Total registros cargados sin deduplicar: {df.shape[0]:,}")

    # Deduplicación por Viaje_Id
    df.drop_duplicates(subset=["Viaje_Id"], keep="first", inplace=True)

    # Conversión eficiente de fechas
    df["Inicio_del_viaje"] = pd.to_datetime(
        df["Inicio_del_viaje"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df["Fin_del_viaje"] = pd.to_datetime(
        df["Fin_del_viaje"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    # Filtro temporal
    df = df[
        (df["Inicio_del_viaje"] >= FECHA_INICIO) & 
        (df["Inicio_del_viaje"] <= FECHA_FIN)
    ]

    # Duración en minutos y filtros de calidad (1 min a 24 horas)
    duracion = (df["Fin_del_viaje"] - df["Inicio_del_viaje"]).dt.total_seconds() / 60.0
    df["duracion_minutos"] = duracion
    df = df[
        (df["duracion_minutos"] >= DURACION_MIN_MINUTOS) & 
        (df["duracion_minutos"] <= DURACION_MAX_MINUTOS)
    ]

    # Filtro de red: tanto origen como destino deben estar en estaciones válidas de AMG
    df = df[
        df["Origen_Id"].isin(estaciones_validas) & 
        df["Destino_Id"].isin(estaciones_validas)
    ]

    # Ordenar y resetear índice in-place
    df.sort_values(by="Inicio_del_viaje", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Guardar en data/interim
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    salida = INTERIM_DATA_DIR / "viajes_amg_limpios.parquet"
    df.to_parquet(salida, index=False)
    
    print(f"Guardado exitoso en {salida}")
    print(f"Dimensiones finales: {df.shape}")
    print(f"Uso de memoria: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actualizar-manifiesto", action="store_true",
                        help="Registra los CSV actuales como nueva versión de los datos crudos.")
    parser.add_argument("--forzar-descarga", action="store_true",
                        help="Re-descarga los CSV aunque ya existan en data/raw/.")
    args = parser.parse_args()

    archivos_faltantes = [p for p in ARCHIVOS_CRUDOS if not p.exists()]
    if archivos_faltantes:
        print(f"Faltan {len(archivos_faltantes)} archivo(s) en data/raw/. Iniciando descarga...")
        descargar_datos_crudos(forzar=args.forzar_descarga)
    elif args.forzar_descarga:
        descargar_datos_crudos(forzar=True)

    verificar_datos_crudos(actualizar=args.actualizar_manifiesto)
    procesar_e_integrar_datos(ARCHIVOS_VIAJES_RAW, RUTA_CATALOGO_ESTACIONES)