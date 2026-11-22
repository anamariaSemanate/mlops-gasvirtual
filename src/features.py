"""Variables de entrada y ventanas de 168 h (reemplaza a clustering.py del tutorial).

Cada muestra tiene:
  - pasado: 168 h × [valor normalizado, calendario cíclico, festivo, fin de semana, estación one-hot]
  - futuro: 168 h × [calendario cíclico, festivo, fin de semana] (se conoce de antemano)
  - y: 168 h de valor normalizado a pronosticar (salida directa, sin recursión)
"""
import numpy as np
import pandas as pd

from src.config import HORIZONTE_H, VENTANA_H

CAL_COLS = ["hora_sin", "hora_cos", "dow_sin", "dow_cos", "mes_sin", "mes_cos",
            "es_festivo", "es_fin_de_semana"]


def agregar_calendario(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hora = df["datetime"].dt.hour
    df["hora_sin"], df["hora_cos"] = np.sin(2 * np.pi * hora / 24), np.cos(2 * np.pi * hora / 24)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * df["dia_semana"] / 7), np.cos(2 * np.pi * df["dia_semana"] / 7)
    df["mes_sin"], df["mes_cos"] = np.sin(2 * np.pi * (df["mes"] - 1) / 12), np.cos(2 * np.pi * (df["mes"] - 1) / 12)
    return df


def estadisticas_normalizacion(train: pd.DataFrame) -> dict:
    """Media y desviación por estación, calculadas SOLO con entrenamiento."""
    s = train.groupby("estacion")["valor_corregido"].agg(["mean", "std"])
    return {e: {"media": float(r["mean"]), "desv": float(r["std"]) or 1.0} for e, r in s.iterrows()}


def matrices_estacion(g: pd.DataFrame, estacion: str, stats: dict, estaciones: list[str]):
    """Devuelve (valor normalizado, calendario, one-hot) como arreglos alineados por hora."""
    g = agregar_calendario(g)
    st = stats[estacion]
    valor = ((g["valor_corregido"].to_numpy() - st["media"]) / st["desv"]).astype("float32")
    cal = g[CAL_COLS].to_numpy(dtype="float32")
    onehot = np.zeros(len(estaciones), dtype="float32")
    onehot[estaciones.index(estacion)] = 1.0
    return valor, cal, onehot


def construir_muestra(valor, cal, onehot, inicio: int):
    """Muestra cuyo primer hora pronosticada es la posición `inicio`."""
    pasado = np.column_stack([
        valor[inicio - VENTANA_H:inicio],
        cal[inicio - VENTANA_H:inicio],
        np.tile(onehot, (VENTANA_H, 1)),
    ])
    futuro = cal[inicio:inicio + HORIZONTE_H]
    return pasado, futuro


def crear_ventanas(df: pd.DataFrame, stats: dict, estaciones: list[str], paso_h: int = 24,
                   desde=None, hasta=None, solo_lunes: bool = False):
    """Ventanas cuya primera hora pronosticada está en [desde, hasta) y cuyo horizonte
    completo cabe en `df`. Con solo_lunes=True solo arrancan los lunes a las 00:00,
    que es como opera el sistema (planeación semanal).
    Devuelve X_pasado, X_futuro, y, metadatos."""
    Xp, Xf, Y, meta = [], [], [], []
    for estacion, g in df.groupby("estacion"):
        g = g.sort_values("datetime").reset_index(drop=True)
        valor, cal, onehot = matrices_estacion(g, estacion, stats, estaciones)
        fechas = g["datetime"].to_numpy()
        ok = np.zeros(len(g), dtype=bool)
        ok[VENTANA_H:len(g) - HORIZONTE_H + 1] = True
        if desde is not None:
            ok &= fechas >= np.datetime64(desde)
        if hasta is not None:
            ok &= fechas < np.datetime64(hasta)
        if solo_lunes:
            ok &= (g["dia_semana"].to_numpy() == 0) & (g["datetime"].dt.hour.to_numpy() == 0)
            indices = np.flatnonzero(ok)
        else:
            candidatos = np.flatnonzero(ok)
            indices = candidatos[(candidatos - VENTANA_H) % paso_h == 0]
        for i in indices:
            p, f = construir_muestra(valor, cal, onehot, i)
            Xp.append(p); Xf.append(f); Y.append(valor[i:i + HORIZONTE_H])
            meta.append((estacion, pd.Timestamp(fechas[i])))
    return (np.asarray(Xp, "float32"), np.asarray(Xf, "float32"),
            np.asarray(Y, "float32"), pd.DataFrame(meta, columns=["estacion", "inicio"]))
