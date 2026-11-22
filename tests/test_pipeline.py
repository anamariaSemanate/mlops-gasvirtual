"""Pruebas de datos, ventanas, métricas y autonomía."""
import numpy as np
import pandas as pd

from src.autonomia import calcular_autonomia
from src.config import HORIZONTE_H, VENTANA_H
from src.data import COLUMNAS, load_processed_data
from src.features import crear_ventanas, estadisticas_normalizacion
from src.fuente_datos import FuenteReplay
from src.metrics import metricas_semana


def test_datos_anonimizados():
    df = load_processed_data()
    assert set(COLUMNAS) <= set(df.columns)
    assert df["estacion"].str.match(r"^EST_\d{2}$").all()
    assert not {"temperatura_promedio", "presion_promedio", "nombre_festivo"} & set(df.columns)
    assert df["valor_corregido"].notna().all() and (df["valor_corregido"] >= 0).all()


def test_ventanas_forma():
    df = load_processed_data()
    df = df[df["estacion"] == "EST_01"].head(24 * 30)
    stats = estadisticas_normalizacion(df)
    Xp, Xf, Y, meta = crear_ventanas(df, stats, ["EST_01"], paso_h=24)
    assert Xp.shape[1:] == (VENTANA_H, 1 + 8 + 1)
    assert Xf.shape[1:] == (HORIZONTE_H, 8)
    assert Y.shape[1] == HORIZONTE_H and len(meta) == len(Y)


def test_replay_oculta_el_futuro():
    fuente = FuenteReplay()
    est, lunes = fuente.estaciones()[0], fuente.rango()[0]
    entrada = fuente.entrada_modelo(est, lunes)
    assert len(entrada) == VENTANA_H + HORIZONTE_H
    assert entrada["valor_corregido"].iloc[VENTANA_H:].isna().all()
    assert entrada["valor_corregido"].iloc[:VENTANA_H].notna().all()


def test_metricas():
    real = np.full(HORIZONTE_H, 10.0)
    m = metricas_semana(real, real * 0.9)
    assert abs(m["sesgo"] + 0.10) < 1e-9
    assert abs(m["wape_diario"] - 0.10) < 1e-9
    assert abs(m["sesgo_horas"] + 16.8) < 1e-9


def test_autonomia():
    idx = pd.date_range("2022-01-03", periods=HORIZONTE_H, freq="h")
    curva = pd.Series(1.0, index=idx)                   # 24 por día
    r = calcular_autonomia(curva, inventario=72, reserva_dias=1, tiempo_viaje_h=6)
    assert r["autonomia_horas"] == 71                    # se agota en la hora 72 (índice 71)
    assert r["hora_reserva"] == idx[47]                  # quedan 24 = 1 día de reserva
    assert r["despacho_sugerido"] == idx[47] - pd.Timedelta(hours=6)
    assert calcular_autonomia(curva, inventario=1000)["alcanza_horizonte"]
