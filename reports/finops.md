# FinOps: estimación del costo por inferencia P1

## Alcance y datos observados

La [prueba con monitoreo](carga/recursos.md), realizada el 20 de noviembre de 2026,
produjo 1,423 peticiones, sin fallos, a un promedio de **15.9783 peticiones/s** y
una latencia p95 aproximada de 350 ms. Cada petición contenía una sola instancia,
por lo que en esta prueba una petición exitosa equivale a una inferencia. Las
mediciones incluyeron la rampa y fueron locales, durante unos 90 segundos.

La API promedió 285.65 % de CPU (aproximadamente 2.86 núcleos lógicos) y 212.49 MiB
de PSS. MLflow, sin atender las predicciones, promedió 3.72 % de CPU y 2,225.72 MiB
de PSS. PSS es una estimación parcial de memoria de procesos, no el consumo total
del cgroup; los límites y la metodología se detallan en el reporte de recursos.

## Precio de referencia y supuestos

Se modela **una** instancia AWS EC2 `m6i.2xlarge` Linux, bajo demanda, en
`us-east-1` (Norte de Virginia): 8 vCPU, 32 GiB de RAM y **USD 0.384 por hora**.
Sus características se parecen a las 8 CPU y ~31.2 GiB visibles en la máquina
Podman, pero no son hardware equivalente. Precio público de referencia consultado
el 20 de noviembre de 2026 en [Economize](https://www.economize.cloud/resources/aws/pricing/ec2/m6i.2xlarge/)
y contrastado con [AWS Pricing](https://aws-pricing.com/m6i.2xlarge.html), ambos
sitios independientes de AWS; [AWS confirma las características de M6i](https://aws.amazon.com/ec2/instance-types/m6i/).
Los precios cambian por región, fecha y condiciones; confirmar la tarifa vigente
con [AWS](https://aws.amazon.com/ec2/pricing/on-demand/) antes de contratar.

Para estimar, se supone que la API y MLflow caben en esa única instancia, que está
encendida continuamente, que conserva el rendimiento local medido y que todas las
predicciones son exitosas. **No se desplegó ni midió esta instancia en AWS**.
El precio es solo de cómputo: excluye almacenamiento EBS, artefactos, transferencia
de datos, balanceador, monitoreo, respaldos, impuestos, soporte y cualquier servidor
separado, réplica o generador de carga. El costo total real sería mayor al sumar
los componentes que correspondan.

## Cálculo de referencia

```text
Inferencias/hora = 15.9783189 peticiones/s × 1 inferencia/petición × 3,600
                  = 57,521.95 inferencias/hora

Costo de cómputo/inferencia = USD 0.384/hora ÷ 57,521.95 inferencias/hora
                            = USD 0.00000668 por inferencia
                            = USD 6.68 por millón de inferencias
```

La tarifa cubre la instancia completa mientras está encendida; no se multiplica por
el porcentaje de CPU observado. Esta cifra supone mantener **continuamente** la tasa
medida. Es una proyección ilustrativa, no un costo marginal garantizado ni una
medición de facturación en nube.

Con 730 horas/mes, la instancia cuesta **USD 280.32/mes** en cómputo. Si permanece
encendida aun con menos tráfico, el costo unitario aumenta:

| Tasa media asumida | Inferencias/mes estimadas | Cómputo USD/mes | USD por millón |
|---|---:|---:|---:|
| 100 % de la tasa medida (15.9783/s) | 41,991,022 | 280.32 | 6.68 |
| 10 % (1.5978/s) | 4,199,102 | 280.32 | 66.76 |
| 1 % (0.1598/s) | 419,910 | 280.32 | 667.57 |

Estos escenarios representan promedios hipotéticos, no tráficos medidos en producción.
Para el cálculo operativo se debe dividir el costo total del periodo entre el número
de **inferencias exitosas**, no solo entre peticiones: `/predict` acepta lotes, y
una petición con varias instancias produce varias inferencias. Si no hay inferencias,
el costo unitario no está definido aunque la instancia siga generando cargos.

## Oportunidades de optimización

1. **Medir y ajustar el tamaño.** Repetir la prueba en la instancia de nube, con
   generador separado, más duraciones y niveles de tráfico; medir memoria de cgroup,
   CPU, p95/p99, fallos y costo total. La CPU media de ~2.86 núcleos sugiere explorar
   configuraciones menores, pero no demuestra que sostengan la carga ni los picos.
   Reservar margen para el sistema y MLflow; comparar costo por inferencia con una
   latencia y tasa de error aceptables antes de cambiar el tamaño.
2. **Administrar MLflow según su uso.** La API usa una copia empaquetada y no necesita
   MLflow para responder predicciones. Evaluar apagar o programar el servidor cuando
   no haya entrenamiento, registro o consultas, preservando el almacenamiento y
   el acceso del equipo. No se ha medido el ahorro: compartir la misma instancia
   no reduce su tarifa por sí solo; podría permitir una configuración más pequeña
   tras comprobarla.
3. **Probar micro-lotes.** `/predict` admite hasta 1,000 instancias por solicitud.
   Comparar lotes representativos midiendo inferencias/s, latencia por lote y por
   instancia, CPU y errores. El costo por inferencia podría disminuir, pero el
   ahorro y el efecto sobre latencia todavía no están medidos.
4. **Adecuar la disponibilidad al tráfico.** Para cargas intermitentes, evaluar
   encendido programado o escalado automático si el tiempo de arranque y la
   disponibilidad lo permiten; para uso estable, comparar compromisos de capacidad
   con su riesgo de pagar por capacidad no utilizada. Incluir redundancia cuando
   se requiera: más réplicas aumentan el costo.

Registrar durante un periodo representativo el número de inferencias exitosas,
el costo por componente y su etiquetado; calcular después
`costo total del periodo / inferencias exitosas`. Esto incorpora horas ociosas,
errores, lotes y costos compartidos, y permite comparar alternativas con las
mismas condiciones de calidad y disponibilidad.
