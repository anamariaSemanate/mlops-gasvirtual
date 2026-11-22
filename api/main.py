"""API de pronóstico y autonomía (Paso 12 del tutorial).

Endpoints:
  GET  /             estado de la API
  GET  /health       modelo cargado, alias y fuente de datos
  GET  /estaciones   estaciones y lunes disponibles para simular
  POST /pronostico   curva de consumo de 168 h para una estación
  POST /autonomia    curva + hora de agotamiento + fecha sugerida de despacho
  POST /evaluacion   compara el pronóstico con el real (solo en simulación) y
                     actualiza las métricas del modelo que ve Grafana
  GET  /metrics      métricas para Prometheus

Ejecuta:  uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
"""
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.autonomia import calcular_autonomia
from src.config import (MLFLOW_TRACKING_URI, MODEL_ALIAS, MONITORING_DIR, PREDICTIONS_LOG_PATH,
                        REGISTERED_MODEL_NAME, RESERVA_SEGURIDAD_DIAS, TIEMPO_VIAJE_H, VENTANA_H)
from src.fuente_datos import FuenteReplay
from src.metrics import metricas_semana

# ======================== MÉTRICAS PROMETHEUS ========================
# Salud del servicio (igual que en el tutorial)
PRONOSTICOS = Counter("gasvirtual_pronosticos_total", "Pronósticos generados", ["estacion"])
LATENCIA = Histogram("gasvirtual_latencia_pronostico_segundos", "Latencia del pronóstico en segundos")
ERRORES = Counter("gasvirtual_errores_total", "Solicitudes con error", ["endpoint"])
# Salud del modelo (lo que el tutorial llama "drift del modelo")
WAPE = Gauge("gasvirtual_wape_diario", "WAPE diario de la última semana evaluada", ["estacion"])
SESGO = Gauge("gasvirtual_sesgo", "Sesgo de la última semana evaluada (negativo = subestima)", ["estacion"])
SESGO_H = Gauge("gasvirtual_sesgo_horas", "Error del consumo semanal en horas de consumo promedio", ["estacion"])
DERIVA = Gauge("gasvirtual_deriva_nivel", "Consumo de la última semana / media de entrenamiento", ["estacion"])
IMPUTADAS = Gauge("gasvirtual_horas_imputadas", "Horas imputadas por calidad de datos en la ventana de entrada", ["estacion"])
AUTONOMIA = Gauge("gasvirtual_autonomia_horas", "Autonomía calculada en la última consulta", ["estacion"])

estado = {"modelo": None, "fuente": None, "error_carga": None}


def cargar_modelo():
    import mlflow
    import mlflow.pyfunc
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    try:
        estado["fuente"] = FuenteReplay()
        if os.getenv("CARGAR_MODELO", "1") == "0":  # pruebas y CI: sin servidor MLflow
            raise RuntimeError("carga del modelo desactivada (CARGAR_MODELO=0)")
        estado["modelo"] = cargar_modelo()
        print("Modelo cargado correctamente")
    except Exception as e:  # la API arranca igual para poder diagnosticar en /health
        estado["error_carga"] = str(e)
        print("No se pudo cargar el modelo:", e)
    yield


app = FastAPI(title="GasVirtual · API de pronóstico y autonomía", lifespan=lifespan)


class SolicitudPronostico(BaseModel):
    estacion: str = Field(examples=["EST_01"])
    inicio: datetime = Field(description="Primera hora a pronosticar (normalmente un lunes 00:00)",
                             examples=["2022-07-25T00:00:00"])


class SolicitudAutonomia(SolicitudPronostico):
    inventario_dias: float | None = Field(None, description="Inventario actual en días de consumo promedio")
    inventario: float | None = Field(None, description="Inventario actual en las unidades de los datos")
    reserva_dias: float = RESERVA_SEGURIDAD_DIAS
    tiempo_viaje_h: float = TIEMPO_VIAJE_H


def _requiere_modelo():
    if estado["modelo"] is None:
        raise HTTPException(503, f"El modelo no está cargado: {estado['error_carga']}")


def _pronosticar(estacion: str, inicio: datetime):
    _requiere_modelo()
    fuente = estado["fuente"]
    if estacion not in fuente.estaciones():
        raise HTTPException(404, f"Estación desconocida: {estacion}")
    inicio = pd.Timestamp(inicio).tz_localize(None)
    try:
        entrada = fuente.entrada_modelo(estacion, inicio)
    except ValueError as e:
        raise HTTPException(422, str(e))

    t0 = time.time()
    salida = estado["modelo"].predict(entrada)
    latencia = time.time() - t0

    curva = salida.set_index("datetime")["pronostico"]
    PRONOSTICOS.labels(estacion).inc()
    LATENCIA.observe(latencia)
    IMPUTADAS.labels(estacion).set(int(entrada["imputado"].iloc[:VENTANA_H].sum()) if "imputado" in entrada else 0)
    _log(estacion, inicio, curva, latencia)
    return curva, entrada, latencia


def _log(estacion, inicio, curva, latencia):
    fila = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "estacion": estacion,
        "inicio": inicio.isoformat(), "consumo_7d": float(curva.sum()), "latencia_s": latencia,
    }])
    fila.to_csv(PREDICTIONS_LOG_PATH, mode="a", header=not PREDICTIONS_LOG_PATH.exists(), index=False)


def _curva_json(curva: pd.Series, real: pd.Series | None = None):
    df = curva.rename("pronostico").to_frame()
    if real is not None:
        df["real"] = real
    return [{"datetime": t.isoformat(), **{k: round(float(v), 3) for k, v in r.items()}}
            for t, r in df.iterrows()]


@app.get("/")
def root():
    return {"message": "GasVirtual API activa"}


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": estado["modelo"] is not None,
            "model": REGISTERED_MODEL_NAME, "alias": MODEL_ALIAS,
            "fuente": "replay (SCADA simulado con histórico)", "error_carga": estado["error_carga"]}


@app.get("/estaciones")
def estaciones():
    fuente = estado["fuente"] or FuenteReplay()
    return {"estaciones": fuente.estaciones(), "lunes_disponibles": [t.isoformat() for t in fuente.rango()]}


@app.post("/pronostico")
def pronostico(req: SolicitudPronostico):
    try:
        curva, _, latencia = _pronosticar(req.estacion, req.inicio)
    except HTTPException:
        ERRORES.labels("pronostico").inc()
        raise
    real = estado["fuente"].real(req.estacion, pd.Timestamp(req.inicio).tz_localize(None))
    return {"estacion": req.estacion, "consumo_7d": float(curva.sum()),
            "latencia_segundos": latencia, "curva": _curva_json(curva, real)}


@app.post("/autonomia")
def autonomia(req: SolicitudAutonomia):
    try:
        curva, entrada, _ = _pronosticar(req.estacion, req.inicio)
    except HTTPException:
        ERRORES.labels("autonomia").inc()
        raise
    if req.inventario is not None:
        inventario = req.inventario
    elif req.inventario_dias is not None:
        inventario = req.inventario_dias * float(entrada["valor_corregido"].iloc[:VENTANA_H].mean() * 24)
    else:
        raise HTTPException(422, "Indica inventario o inventario_dias")

    r = calcular_autonomia(curva, inventario, req.reserva_dias, req.tiempo_viaje_h)
    if r["autonomia_horas"] is not None:
        AUTONOMIA.labels(req.estacion).set(r["autonomia_horas"])
    iso = lambda t: t.isoformat() if t is not None else None
    return {
        "estacion": req.estacion, "inventario_inicial": inventario,
        "consumo_promedio_dia": r["consumo_promedio_dia"], "reserva_seguridad": r["reserva_seguridad"],
        "autonomia_horas": r["autonomia_horas"], "hora_reserva": iso(r["hora_reserva"]),
        "hora_agotamiento": iso(r["hora_agotamiento"]), "despacho_sugerido": iso(r["despacho_sugerido"]),
        "alcanza_horizonte": r["alcanza_horizonte"],
        "curva": _curva_json(curva),
        "inventario_restante": [round(float(v), 3) for v in r["inventario_restante"]],
    }


@app.post("/evaluacion")
def evaluacion(req: SolicitudPronostico):
    curva, entrada, _ = _pronosticar(req.estacion, req.inicio)
    real = estado["fuente"].real(req.estacion, pd.Timestamp(req.inicio).tz_localize(None))
    if real is None:
        raise HTTPException(422, "Todavía no se conoce el consumo real de ese horizonte")
    m = metricas_semana(real.to_numpy(), curva.to_numpy())
    WAPE.labels(req.estacion).set(m["wape_diario"])
    SESGO.labels(req.estacion).set(m["sesgo"])
    SESGO_H.labels(req.estacion).set(m["sesgo_horas"])
    media_train = estado["fuente"].df.query("estacion == @req.estacion and datetime < @estado['fuente'].inicio_test")["valor_corregido"].mean()
    DERIVA.labels(req.estacion).set(float(entrada["valor_corregido"].iloc[:VENTANA_H].mean() / media_train))
    return {"estacion": req.estacion, "inicio": req.inicio.isoformat(), **{k: float(v) for k, v in m.items()}}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
