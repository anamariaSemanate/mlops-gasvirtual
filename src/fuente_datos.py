"""Fuentes de datos para la operación.

No hay acceso al SCADA, así que la operación se simula con el histórico
(FuenteReplay) o con un archivo exportado (FuenteArchivo). Ambas cumplen la misma
interfaz; conectar el SCADA real sería agregar una FuenteSCADA con la consulta SQL
sin cambiar la API ni el modelo.
"""
from abc import ABC, abstractmethod

import pandas as pd

from src.config import HORIZONTE_H, VENTANA_H
from src.data import control_calidad, fecha_corte_test, load_processed_data


class FuenteDatos(ABC):
    @abstractmethod
    def estaciones(self) -> list[str]: ...

    @abstractmethod
    def entrada_modelo(self, estacion: str, inicio: pd.Timestamp) -> pd.DataFrame:
        """168 h de historia antes de `inicio` + 168 h futuras (calendario) desde `inicio`."""

    def real(self, estacion: str, inicio: pd.Timestamp) -> pd.Series | None:
        """Consumo real del horizonte, si ya se conoce (solo en simulación)."""
        return None


class FuenteReplay(FuenteDatos):
    """SCADA simulado: entrega el histórico anonimizado como si llegara en tiempo real."""

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df if df is not None else load_processed_data()
        self.inicio_test = fecha_corte_test(self.df)

    def estaciones(self):
        return sorted(self.df["estacion"].unique())

    def rango(self):
        """Lunes disponibles para simular (periodo de test, horizonte completo)."""
        fin = self.df.groupby("estacion")["datetime"].max().min() - pd.Timedelta(hours=HORIZONTE_H - 1)
        lunes = pd.date_range(self.inicio_test, fin, freq="h")
        return [t for t in lunes if t.dayofweek == 0 and t.hour == 0]

    def _tramo(self, estacion, inicio):
        g = self.df[self.df["estacion"] == estacion]
        desde = inicio - pd.Timedelta(hours=VENTANA_H)
        hasta = inicio + pd.Timedelta(hours=HORIZONTE_H)
        return g[(g["datetime"] >= desde) & (g["datetime"] < hasta)].sort_values("datetime")

    def entrada_modelo(self, estacion, inicio):
        tramo = self._tramo(estacion, inicio).copy()
        if len(tramo) != VENTANA_H + HORIZONTE_H:
            raise ValueError(f"No hay datos completos de {estacion} para {inicio}")
        # el futuro no se conoce en operación: se borra para que el modelo no lo vea
        tramo.loc[tramo["datetime"] >= inicio, "valor_corregido"] = float("nan")
        return tramo

    def real(self, estacion, inicio):
        tramo = self._tramo(estacion, inicio)
        futuro = tramo[tramo["datetime"] >= inicio]
        return futuro.set_index("datetime")["valor_corregido"] if len(futuro) == HORIZONTE_H else None


class FuenteArchivo(FuenteDatos):
    """Archivo exportado (CSV o parquet) con las columnas de data.COLUMNAS.
    Debe traer al menos 168 h de historia y las 168 h futuras con su calendario."""

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        self.df = control_calidad(df)

    def estaciones(self):
        return sorted(self.df["estacion"].unique())

    def entrada_modelo(self, estacion, inicio):
        return FuenteReplay(self.df).entrada_modelo(estacion, inicio)
