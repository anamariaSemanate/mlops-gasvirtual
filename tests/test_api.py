"""Pruebas de la API (Paso 20 del tutorial). No requieren MLflow ni el modelo entrenado."""
from fastapi.testclient import TestClient

from api.main import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_estaciones():
    with TestClient(app) as client:
        datos = client.get("/estaciones").json()
        assert len(datos["estaciones"]) == 5
        assert len(datos["lunes_disponibles"]) > 0


def test_metrics():
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "gasvirtual_pronosticos_total" in response.text
