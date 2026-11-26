# Model Card — demanda horaria MiBici P1

## Identificación y alcance

- **Modelo:** `mibici-equipo33-p1-demanda`, versión registrada `1`,
  `HistGradientBoostingRegressor`. El alias `champion` apuntaba a esta versión
  en la base local consultada el 25 de noviembre de 2026; compruébalo en el
  Registry activo antes de desplegar.
- **Run:** `606849422b6941c38df8c9039bdcff5d`, experimento
  `mibici-equipo33`; commit `c59e6e0c7fa67f9db806ee5a2ec2eff3cd185695`,
  registrado con `git_dirty=False`. Datos: `mibici-datos-abiertos-2026_04-2026_09`.
  Ver [linaje y selección](parte-c-model-registry.md) y
  [manifiesto de datos](../references/manifiesto_datos.json).
- **Propósito:** estimar el número de viajes que empiezan en una estación durante
  la hora a pronosticar, para apoyar la planificación de rebalanceo. La salida es
  una estimación continua, no una cantidad de bicicletas disponibles.
- **Ámbito:** estaciones de MiBici dentro del Área Metropolitana de Guadalajara,
  según los filtros del proyecto. No se ha validado para otras redes o zonas.
- **No usar como única base** para decisiones automáticas de redistribución, ni
  para inferir ocupación real, evaluar personas o garantizar disponibilidad.

## Datos y variables

Los datos proceden del [portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/).
Se limpiaron viajes y estaciones, se eliminó duplicación por viaje y se generó una
malla estación-hora con ceros donde no se observaron salidas. El entrenamiento
comprende abril–agosto de 2026; septiembre es la prueba. Se descartan las primeras
168 horas por falta de historia. Detalles en [datos y pipeline](../README.md#datos)
y [features](../src/features.py).

La entrada contiene hora y día de semana cíclicos, indicador de fin de semana,
rezagos de salidas de 1, 2, 3, 24 y 168 horas y media móvil de las seis horas
anteriores. El cliente debe aportar esas variables para la estación y hora
pronosticadas; la API no calcula los rezagos. No se utilizan identificador,
género ni año de nacimiento de las personas como variables del modelo. Los CSV
crudos contienen otros campos y deben tratarse según las políticas aplicables.
El run histórico registra el hash abreviado de los datos crudos, pero no los hashes de
los Parquet preparados; no se puede demostrar que una regeneración con el pipeline actual
sea idéntica a los datos usados para entrenar la V1.

La API acepta de 1 a 1,000 instancias por petición, valida el esquema y recorta
predicciones negativas a cero; devuelve nombre y versión del modelo. El contrato
está en [la documentación de la API](../README.md#2-api-rest-con-fastapi).

## Evaluación y evidencia

En septiembre de 2026, sobre 337,680 filas de 469 estaciones, la V1 obtuvo
RMSE **1.4810**, MAE **0.8373** y R² **0.6917**. Son métricas agregadas de un
mes retrospectivo, no una garantía de desempeño por estación o en producción.

La comparación de las versiones 1 y 2 sobre las mismas filas obtuvo RMSE 1.4810
y 1.4848, respectivamente. El cambio V2−V1 fue +0.0038, con intervalo bootstrap
pareado por día al 95 % de [−0.0014, +0.0119]. No hay mejora concluyente; la V2
además registra `git_dirty=True` y no debe promoverse. Esta evaluación no es un
experimento con tráfico aleatorizado. Ver [resultados A/B](monitoring/ab_test_p1.json).
Septiembre también fue usado para evaluar los baselines; una promoción nueva debe
evaluarse con un periodo posterior independiente.

## Monitoreo y operación

El [reporte Evidently](monitoring/drift_p1.html) compara abril–agosto con septiembre
de 2026 y detectó drift en 7 de 12 variables, incluido el target; el
[JSON](monitoring/drift_p1.json) contiene los métodos y umbrales. El drift es una
señal para investigar, no demuestra por sí solo pérdida de calidad. La API no
almacena solicitudes: este análisis es retrospectivo, no monitoreo en vivo, y
medir drift o error del target requiere disponer de resultados observados.

Ante cambios, revisar calidad, errores por estación y periodo y los resultados de
una candidata; entrenarla desde un commit identificable y árbol limpio, evaluarla
en un periodo independiente y justificar la promoción antes de mover el alias
`champion`. Ninguna alerta ni promoción es automática. El procedimiento y sus
limitaciones están en [monitoreo y gobernanza](../README.md#monitoreo-y-gobernanza-p1).

El [análisis FinOps](finops.md) estima USD 6.68 por millón de inferencias bajo
una tasa continua y supuestos de cómputo específicos; no es una factura real ni
incluye todos los costos. Revisar la tarifa, el tráfico y los demás componentes
antes de usar esa estimación operativamente.

## Limitaciones y riesgos

- La demanda observada depende de la disponibilidad de bicicletas y anclajes;
  no equivale necesariamente a demanda insatisfecha.
- Clima, eventos, feriados y cambios de servicio no son variables explícitas.
  Los cambios de red y patrones de viaje pueden reducir la validez del modelo.
- No se ha documentado una evaluación de equidad o error desagregado por zona,
  estación o población. El rendimiento agregado puede ocultar desigualdades;
  revisar esos segmentos antes de apoyar decisiones operativas.
- Las métricas, el drift, los costos y el alias de esta ficha son evidencia
  histórica. Actualizarla junto con el linaje y los resultados al cambiar modelo,
  datos o condiciones de operación.
