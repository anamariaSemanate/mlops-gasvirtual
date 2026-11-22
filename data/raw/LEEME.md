# Datos anonimizados

`series_anonimizadas.parquet`: consumo horario de 5 estaciones de gas virtual.

| Columna | Descripción |
|---|---|
| estacion | Código anónimo (EST_01 … EST_05) |
| datetime | Fecha y hora (desplazada; conserva el día de la semana) |
| valor_corregido | Volumen corregido por hora, escalado con un factor secreto por estación |
| dia_semana, mes | Calendario de la fecha real (0 = lunes) |
| es_festivo, es_fin_de_semana | Indicadores 0/1 de la fecha real (festivos de Colombia) |

Generado con `scripts/anonimizar_datos.py`. La clave para revertir no se publica.
Las métricas relativas (WAPE, sesgo %) son idénticas a las de los datos reales.
