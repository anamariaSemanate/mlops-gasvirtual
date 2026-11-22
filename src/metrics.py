"""Métricas del pronóstico de 168 h. Todas son relativas, así que dan igual
con datos anonimizados (escalados) que con datos reales."""
import numpy as np
import pandas as pd

from src.config import HORIZONTE_H


def metricas_semana(real: np.ndarray, pred: np.ndarray) -> dict:
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    total = real.sum()
    dias_r = real.reshape(-1, 24).sum(axis=1)
    dias_p = pred.reshape(-1, 24).sum(axis=1)
    sesgo = (pred.sum() - total) / total if total else np.nan
    return {
        "wape_horario": np.abs(real - pred).sum() / total if total else np.nan,
        "wape_diario": np.abs(dias_r - dias_p).sum() / total if total else np.nan,
        "sesgo": sesgo,
        # error del consumo semanal expresado en horas de consumo promedio
        "sesgo_horas": sesgo * HORIZONTE_H,
        "max_vs_real": pred.max() / real.max() if real.max() else np.nan,
    }


def resumir(filas: list[dict]) -> dict:
    """Agrega las métricas de muchas semanas: WAPE ponderado y % de semanas subestimadas."""
    df = pd.DataFrame(filas)
    total = df["total_real"].sum()
    return {
        "wape_horario": float((df["wape_horario"] * df["total_real"]).sum() / total),
        "wape_diario": float((df["wape_diario"] * df["total_real"]).sum() / total),
        "sesgo": float((df["sesgo"] * df["total_real"]).sum() / total),
        "sesgo_horas": float((df["sesgo"] * df["total_real"]).sum() / total * HORIZONTE_H),
        "pct_semanas_subestimadas": float((df["sesgo"] < 0).mean()),
        "max_vs_real": float(df["max_vs_real"].mean()),
        "n_semanas": int(len(df)),
    }
