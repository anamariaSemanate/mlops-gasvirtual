"""Cálculo de autonomía y fecha sugerida de despacho a partir de la curva pronosticada."""
import pandas as pd

from src.config import RESERVA_SEGURIDAD_DIAS, TIEMPO_VIAJE_H


def calcular_autonomia(curva: pd.Series, inventario: float,
                       reserva_dias: float = RESERVA_SEGURIDAD_DIAS,
                       tiempo_viaje_h: float = TIEMPO_VIAJE_H) -> dict:
    """
    curva: consumo pronosticado por hora (índice = datetime).
    inventario: volumen disponible al inicio de la curva (mismas unidades que la curva).
    La reserva de seguridad equivale a `reserva_dias` de consumo promedio pronosticado.
    """
    consumo_dia = float(curva.mean() * 24)
    reserva = reserva_dias * consumo_dia
    restante = inventario - curva.cumsum()

    bajo_reserva = restante[restante <= reserva]
    agotado = restante[restante <= 0]
    hora_reserva = bajo_reserva.index[0] if len(bajo_reserva) else None
    hora_agotamiento = agotado.index[0] if len(agotado) else None
    despacho = hora_reserva - pd.Timedelta(hours=tiempo_viaje_h) if hora_reserva is not None else None
    inicio = curva.index[0]

    return {
        "consumo_promedio_dia": consumo_dia,
        "reserva_seguridad": reserva,
        "autonomia_horas": (hora_agotamiento - inicio) / pd.Timedelta(hours=1) if hora_agotamiento is not None else None,
        "hora_reserva": hora_reserva,
        "hora_agotamiento": hora_agotamiento,
        "despacho_sugerido": max(despacho, inicio) if despacho is not None else None,
        "alcanza_horizonte": hora_reserva is None,
        "inventario_restante": restante,
    }
