# MLOps GasVirtual · Pronóstico de consumo y autonomía de estaciones de gas virtual

Proyecto MLOps de punta a punta, basado en el tutorial *"Cómo construir un proyecto MLOps completo con Python, MLflow, FastAPI, Streamlit y Grafana"* (J. M. Núñez Velasco), adaptado a un problema real: pronosticar el consumo horario de gas de 5 estaciones a 7 días y calcular cuándo enviar el camión de reabastecimiento.

## Arquitectura

```
Datos anonimizados ─► src/data ─► src/features ─► src/train ──► MLflow (tracking + registry)
                                                                     │ alias "champion"
SCADA simulado (replay) ─► FastAPI /pronostico /autonomia ◄──────────┘
                                │  /metrics
              Streamlit ◄───────┤
                                └──► Prometheus ──► Grafana
```

| Componente | Uso en este proyecto |
|---|---|
| **MLflow** | Registra cada experimento (ingenuo estacional y LSTM global con 3 semillas): WAPE horario y diario, sesgo, % de semanas subestimadas, gráficas. El mejor modelo recibe el alias `champion`. |
| **FastAPI** | Sirve el pronóstico de 168 h y el cálculo de autonomía; expone métricas para Prometheus. |
| **Streamlit** | Tablero del operador: estación, semana, inventario → autonomía y despacho sugerido. |
| **Prometheus + Grafana** | Salud del servicio (latencia, conteo, errores) y **salud del modelo** (WAPE, sesgo, deriva de nivel, calidad de datos) por estación. |
| **Docker Compose** | Levanta Prometheus y Grafana con el tablero ya configurado. |
| **GitHub Actions** | Verifica que no haya datos privados, sintaxis, preprocesamiento y pruebas. |

### Diferencias con el tutorial

- **No hay `clustering.py`**: el tutorial lo usa para crear la etiqueta; aquí la etiqueta es el consumo futuro.
- **`features.py`** arma ventanas de 168 h de historia + calendario de las 168 h futuras.
- **Partición por fecha** (últimos 60 días de test) en vez de `train_test_split` aleatorio, y evaluación *walk-forward*: un pronóstico por estación cada lunes.
- **Criterio del champion**: menor WAPE diario entre los modelos que no subestiman más de 3 %. Subestimar el consumo lleva a desabastecimiento.
- **SCADA simulado**: no hay acceso al SCADA. `src/fuente_datos.py` define una interfaz con `FuenteReplay` (histórico) y `FuenteArchivo` (archivo exportado); conectar el SCADA real sería agregar una `FuenteSCADA`.
- **`scripts/`**: pasos que no son parte del pipeline (anonimización y simulación de operación).

## Datos y anonimización

Los datos originales son privados de la empresa operadora y **no están en este repositorio**. `data/raw/series_anonimizadas.parquet` se generó con `scripts/anonimizar_datos.py`:

1. Estaciones → códigos `EST_01`…`EST_05` asignados al azar.
2. Volumen × factor secreto por estación: no se conocen los consumos reales, pero WAPE, sesgo y la forma de la curva son idénticos.
3. Fechas desplazadas un número secreto de años de 364 días: se conserva el día de la semana; festivos y fin de semana se calcularon con las fechas reales.
4. Se eliminaron temperatura, presión y nombres de festivos.

La clave para revertir (`clave_anonimizacion.json`) y la lista de estaciones que se combinan (`estaciones_combinar.json`) son archivos locales: están en `.gitignore` y el CI falla si aparecen en el repositorio. El script no contiene nombres reales de estaciones; los lee de los archivos de origen. Detalle de columnas en [`data/raw/LEEME.md`](data/raw/LEEME.md).

## Estructura

```
├── api/main.py                  API FastAPI
├── app/streamlit_app.py         Tablero del operador
├── data/
│   ├── raw/                     Datos anonimizados
│   ├── processed/               Datos con control de calidad (se generan)
│   └── monitoring/              Log de predicciones de la API
├── docker/                      Prometheus y Grafana (datasource + tablero)
├── reports/figures, metrics/    Gráficas y tablas de los experimentos
├── scripts/
│   ├── anonimizar_datos.py      Anonimización (se corre una vez, con datos privados)
│   └── simular_operacion.py     SCADA simulado: recorre las semanas de test
├── src/
│   ├── config.py                Configuración centralizada
│   ├── data.py                  Carga, control de calidad y corte de test
│   ├── features.py              Calendario cíclico y ventanas de 168 h
│   ├── metrics.py               WAPE, sesgo, sesgo en horas
│   ├── modelo.py                Red LSTM y modelo pyfunc para MLflow
│   ├── train.py                 Entrenamiento y tracking
│   ├── register_model.py        Registro del champion
│   ├── predict.py               Prueba local
│   ├── fuente_datos.py          Replay / archivo (sustituye al SCADA)
│   └── autonomia.py             Autonomía y despacho sugerido
├── tests/                       Pruebas (pytest)
└── docker-compose.yml
```

## Cómo ejecutarlo (Windows / PowerShell)

El entorno se crea **fuera de OneDrive** y en una ruta corta: TensorFlow tiene archivos con rutas muy largas y Windows no admite más de 260 caracteres (además, OneDrive intentaría sincronizar ~2 GB).

```powershell
py -3.12 -m venv C:\venvs\mlops-gasvirtual
C:\venvs\mlops-gasvirtual\Scripts\activate
pip install -r requirements.txt
```

TensorFlow requiere Python 3.12 o anterior. En VS Code, las tareas de `.vscode/tasks.json` (Terminal → Run Task) ya usan ese entorno.

**Terminal 1 · MLflow**
```powershell
mlflow server --host 127.0.0.1 --port 5000
```

**Terminal 2 · pipeline** (una vez)
```powershell
python -m src.data             # control de calidad
python -m src.train            # entrena y registra experimentos (≈10-20 min en CPU)
python -m src.register_model   # elige el champion
python -m src.predict          # prueba local
```

**Terminal 2 · API**
```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
Documentación interactiva: http://127.0.0.1:8000/docs

**Terminal 3 · Streamlit**
```powershell
streamlit run app/streamlit_app.py
```
http://localhost:8501

**Terminal 4 · Monitoreo**
```powershell
docker compose up -d
python scripts/simular_operacion.py --pausa 10
```
- Prometheus: http://localhost:9090 (consulta `gasvirtual_wape_diario`)
- Grafana: http://localhost:3000 (admin / admin) → carpeta *GasVirtual* → tablero *Monitoreo del pronóstico*

**Pruebas**
```powershell
pytest -q
```

## Métricas de Prometheus

| Métrica | Qué mide |
|---|---|
| `gasvirtual_pronosticos_total{estacion}` | Pronósticos generados |
| `gasvirtual_latencia_pronostico_segundos` | Latencia del modelo |
| `gasvirtual_errores_total{endpoint}` | Solicitudes con error |
| `gasvirtual_wape_diario{estacion}` | Error diario de la última semana evaluada |
| `gasvirtual_sesgo{estacion}` | Sesgo (negativo = subestima) |
| `gasvirtual_sesgo_horas{estacion}` | Error del consumo semanal en horas de consumo |
| `gasvirtual_deriva_nivel{estacion}` | Consumo reciente / media de entrenamiento |
| `gasvirtual_horas_imputadas{estacion}` | Calidad de datos en la ventana de entrada |
| `gasvirtual_autonomia_horas{estacion}` | Autonomía de la última consulta |
