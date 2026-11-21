# Uso de recursos durante la prueba de carga

## Condiciones

El 20 de noviembre de 2026, de 17:56:31 a 17:58:00 (UTC-6), se repitió el escenario
de [Locust](../../load_tests/locustfile.py): 20 usuarios, rampa de 2 usuarios/s,
90 segundos incluyendo la rampa y una instancia por petición. Los servicios `api`
y `mlflow` se levantaron juntos con Compose y Podman. La API sirvió el modelo real
`mibici-equipo33-p1-demanda` versión 1. MLflow permaneció disponible, pero la API
no le consulta durante la inferencia.

El cliente y los contenedores compartieron un equipo Windows 11 con Intel i7-11850H.
Podman 5.8.3 usó WSL y reportó 8 CPU; `/proc/meminfo` dentro de la máquina mostró
aproximadamente 31.2 GiB de memoria total. La configuración de la máquina indicaba
2 GiB, pero esa cifra no coincide con la memoria observada en WSL y no se asumió como
límite efectivo del contenedor.

## Resultados durante la ventana de carga

| Contenedor | CPU media | CPU máxima | PSS medio | PSS máximo |
|---|---:|---:|---:|---:|
| API | 285.65 % | 339.04 % | 212.49 MiB | 213.08 MiB |
| MLflow | 3.72 % | 6.64 % | 2,225.72 MiB | 2,225.84 MiB |

La CPU procede de 18 muestras por contenedor de `podman stats`, con intervalos de
aproximadamente 5 s; 100 % equivale al uso de un núcleo lógico, por lo que puede
superar 100 %. Se excluyó la primera lectura acumulada y se seleccionaron únicamente
las muestras con fecha dentro de la ventana de Locust. Las medias son medias de
muestras, no promedios ponderados por tiempo.

La memoria es la suma del **PSS** de los procesos visibles de cada contenedor,
tomada de `/proc/<pid>/smaps_rollup`: 9 muestras de la API y 10 de MLflow durante
la carga. PSS reparte las páginas compartidas entre procesos; no incluye toda la
memoria contabilizable al cgroup, como caché de página o memoria del kernel.
No se debe interpretar como el límite de memoria ni como una medición equivalente
a `docker stats`. Las muestras de CPU y PSS no son simultáneas; la lectura PSS se
hizo mediante `podman exec`, excluyendo el proceso auxiliar de lectura.

En este entorno, `podman stats` mostró `0B / 33.52GB` para ambos contenedores y no
existía `/sys/fs/cgroup/memory.current`. Por ello esos valores no se usaron como
consumo real. La consulta PSS y la herramienta de carga también consumen recursos
y pueden afectar las mediciones. La API y MLflow estaban en la misma máquina.

La ejecución generó **1,423 peticiones**, **0 fallos**, **15.98 peticiones/s**
de promedio y **350 ms de p95**. Son resultados de esta ejecución, no los de la
[prueba anterior](reporte.md). No establecen capacidad máxima ni un límite de
memoria para producción.

Evidencia: [muestras de CPU y memoria informada por Podman](recursos_cpu.csv),
[muestras PSS](recursos_memoria.csv),
[resumen Locust](locust-recursos_stats.csv),
[historial Locust](locust-recursos_stats_history.csv),
[fallos](locust-recursos_failures.csv) y [excepciones](locust-recursos_exceptions.csv).
Los CSV de recursos también contienen muestras fuera de la ventana de carga; la
tabla anterior solo incluye las tomadas entre 23:56:31 y 23:58:00.226 UTC.

## Repetir la observación

Levanta Compose con el modelo real y ejecuta la prueba Locust como se explica en
el [reporte anterior](reporte.md). En otra terminal, identifica los nombres con
`podman ps` y observa ambos contenedores durante la carga:

```powershell
podman stats --no-reset --interval 5 <nombre-api> <nombre-mlflow>
```

En sistemas donde la columna de memoria sea válida, úsala como medición del cgroup.
Si, como aquí, no está disponible, se puede obtener una muestra PSS por contenedor
con Python dentro de la imagen. Reemplaza el nombre por el mostrado en `podman ps`:

```powershell
$codigo = "import os,pathlib; print(sum(int(next(line for line in p.read_text().splitlines() if line.startswith('Pss:')).split()[1]) for p in pathlib.Path('/proc').glob('[0-9]*/smaps_rollup') if int(p.parent.name) != os.getpid()))"
podman exec <nombre-api> python -c $codigo
podman exec <nombre-mlflow> python -c $codigo
```

El resultado está en KiB; dividir entre 1024 para obtener MiB. Si no se puede leer
`smaps_rollup`, la orden falla: no debe sustituirse el resultado por cero.
