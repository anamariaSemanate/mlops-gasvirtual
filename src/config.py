"""Configuración centralizada del proyecto (Paso 5 del tutorial).

Rutas, nombres de MLflow y parámetros del pronóstico en un solo lugar.
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- rutas
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MONITORING_DIR = DATA_DIR / "monitoring"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

RAW_DATA_PATH = RAW_DATA_DIR / "series_anonimizadas.parquet"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "series_procesadas.parquet"
PREDICTIONS_LOG_PATH = MONITORING_DIR / "api_predicciones_log.csv"

# ---------------------------------------------------------------- MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "gasvirtual-pronostico-168h"
REGISTERED_MODEL_NAME = "gasvirtual-pronosticador"
MODEL_ALIAS = "champion"

# ---------------------------------------------------------------- pronóstico
VENTANA_H = 168          # horas de historia que ve el modelo (1 semana)
HORIZONTE_H = 168        # horas a pronosticar (7 días)
TEST_DIAS = 60           # últimos días reservados para evaluación
VAL_FRACCION = 0.15      # fracción final del entrenamiento usada como validación
PASO_MUESTRAS_H = 24     # cada cuántas horas se toma una ventana de entrenamiento
SEMILLAS = [42, 7, 123]
EPOCAS = int(os.getenv("EPOCAS", "30"))

# Criterio para elegir el modelo "champion": menor WAPE diario entre los que
# no subestimen más que este límite (subestimar = riesgo de desabastecimiento).
SESGO_MINIMO = -0.03

# ---------------------------------------------------------------- autonomía
# Con datos anonimizados los volúmenes están escalados, por eso el inventario
# también se expresa en "días de consumo promedio" en la interfaz.
RESERVA_SEGURIDAD_DIAS = 0.8   # inventario mínimo antes de considerar agotada la estación
TIEMPO_VIAJE_H = 6             # horas que tarda el camión en llegar

# ---------------------------------------------------------------- servicios
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
