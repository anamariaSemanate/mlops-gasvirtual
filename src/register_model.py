"""Registra el mejor modelo y le asigna el alias "champion" (Paso 10 del tutorial).

Criterio (adaptado al riesgo operativo): menor WAPE diario entre los runs cuyo sesgo
no sea más negativo que SESGO_MINIMO. Subestimar el consumo hace que el sistema
crea que el gas dura más de lo real, que es el error que lleva a desabastecimiento.
Si ningún run cumple, se toma el de menor WAPE diario y se avisa.

Ejecuta:  python -m src.register_model
"""
import mlflow
from mlflow.tracking import MlflowClient

from src.config import (EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_ALIAS,
                        REGISTERED_MODEL_NAME, SESGO_MINIMO)


def register_best_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError("No existe el experimento. Ejecuta primero: python -m src.train")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.sesgo >= {SESGO_MINIMO}",
        order_by=["metrics.wape_diario ASC"], max_results=1,
    )
    if not runs:
        print(f"Ningún run cumple sesgo >= {SESGO_MINIMO:.0%}; se elige solo por WAPE diario.")
        runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                                  order_by=["metrics.wape_diario ASC"], max_results=1)
    if not runs:
        raise ValueError("No hay runs disponibles")

    best = runs[0]
    model_uri = best.data.tags.get("model_uri", f"runs:/{best.info.run_id}/model")
    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, result.version)

    m = best.data.metrics
    print("Modelo registrado correctamente")
    print("Run:", best.info.run_name, "|", best.info.run_id)
    print(f"WAPE diario: {m['wape_diario']:.3%} | sesgo: {m['sesgo']:+.2%}")
    print(f"Modelo: {REGISTERED_MODEL_NAME} | versión {result.version} | alias {MODEL_ALIAS}")


if __name__ == "__main__":
    register_best_model()
