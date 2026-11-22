"""Prueba local de predicción con el modelo "champion" (Paso 11 del tutorial).

Ejecuta:  python -m src.predict
"""
import mlflow
import mlflow.pyfunc

from src.config import MLFLOW_TRACKING_URI, MODEL_ALIAS, REGISTERED_MODEL_NAME
from src.fuente_datos import FuenteReplay
from src.metrics import metricas_semana


def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")


def predict_estacion():
    model = load_model()
    fuente = FuenteReplay()
    estacion, inicio = fuente.estaciones()[0], fuente.rango()[-1]

    curva = model.predict(fuente.entrada_modelo(estacion, inicio))
    real = fuente.real(estacion, inicio)
    print(f"Estación {estacion} · pronóstico desde {inicio:%Y-%m-%d %H:%M}")
    print(curva.head())
    print("Consumo pronosticado 7 días:", round(curva["pronostico"].sum(), 1))
    if real is not None:
        m = metricas_semana(real.to_numpy(), curva["pronostico"].to_numpy())
        print(f"Consumo real 7 días: {real.sum():.1f} | WAPE diario {m['wape_diario']:.2%} | sesgo {m['sesgo']:+.2%}")


if __name__ == "__main__":
    predict_estacion()
