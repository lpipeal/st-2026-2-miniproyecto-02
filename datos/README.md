# datos

Esta carpeta contiene la serie mensual del peaje Siberia del miniproyecto 1
y el calendario de festivos que usa el SARIMAX.

| Campo | Valor |
|---|---|
| Archivo | [trafico_vehicular_peaje_siberia.csv](trafico_vehicular_peaje_siberia.csv) |
| Qué mide | Vehículos de las categorías tarifarias III a VII que pasan por el peaje Siberia, en el corredor Bogotá–Medellín |
| Fuente | Agencia Nacional de Infraestructura (ANI), [Tráfico Vehicular ANI en Datos Abiertos Colombia](https://www.datos.gov.co/d/8yi9-t44c) |
| Licencia | Creative Commons Atribución-CompartirIgual 4.0 Internacional (CC BY-SA 4.0) |
| Frecuencia | Mensual |
| Periodo | Enero de 2014 a mayo de 2026, con 149 observaciones |
| Columna de fecha | `mes`, primer día del mes |
| Columna que se pronostica | `total`, número de vehículos en el mes |

## Cómo se construyó y preparó la serie

En el miniproyecto 1 se consultó la API Socrata para sumar
`cantidadtrafico` por mes, filtrando el peaje Siberia y las categorías
III a VII. Se trajo el mismo CSV a este repositorio.

```text
https://www.datos.gov.co/resource/8yi9-t44c.csv
  ?$select=date_trunc_ym(desde) as mes, sum(cantidadtrafico) as total
  &$where=peaje='SIBERIA' and categoriatarifa in('III','IV','V','VI','VII')
  &$group=mes
  &$order=mes
```

El archivo tiene una fila por mes, sin huecos ni fechas duplicadas. Los
totales van de 60.922 a 112.471 vehículos. El diagnóstico de 1,5 veces el
rango intercuartílico detectó cero observaciones atípicas. El guion conserva
los valores registrados.

## Calendario de festivos

[festivos_colombia.csv](festivos_colombia.csv) contiene el número de fechas
festivas nacionales de Colombia por mes. Se exportó con la misma función
del miniproyecto 1, usando `holidays` 0.107, país `CO` y
`observed=True` para incluir los traslados del calendario. Tiene las
columnas `mes` y `festivos` y cubre 161 meses, de enero de 2014 a mayo
de 2027. Los meses sin festivos tienen valor cero.

En cada origen, el SARIMAX toma de este archivo los festivos de los meses
de entrenamiento y de los 12 meses que pronostica. El tráfico observado
del horizonte se usa para evaluar el pronóstico. Para ajustar el modelo,
el guion divide el tráfico por 10.000 y devuelve los pronósticos en
vehículos.

## Resultado de referencia

En la corrida guardada en este repositorio, el **SARIMAX con festivos**
obtuvo el menor error de la comparación, con un RMSE de **4.131,63
vehículos**, un MAE de **3.466,94 vehículos** y un MAPE de **3,33 %**.
La tabla completa está en [resultados/metricas.csv](../resultados/metricas.csv)
y el análisis está en [docs/informe.md](../docs/informe.md).
