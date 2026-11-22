"""Modelo empaquetado para MLflow (pyfunc).

Envuelve la red y la normalización en un solo objeto, para que la API solo tenga
que hacer `model.predict(df)` igual que en el tutorial.

Entrada de predict(): DataFrame con las columnas de data.COLUMNAS y, por cada
estación, las últimas 336 filas horarias consecutivas:
  - 168 filas de historia (valor_corregido conocido)
  - 168 filas futuras (solo se usa su calendario; valor_corregido puede ser NaN)
Salida: DataFrame [estacion, datetime, pronostico] con 168 filas por estación.
"""
import json

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.config import HORIZONTE_H, VENTANA_H
from src.features import construir_muestra, matrices_estacion


def construir_red(n_pasado: int, n_futuro: int, semilla: int, unidades: int = 64):
    """LSTM global con salida directa de 168 h (sin recursión)."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(semilla)
    L = tf.keras.layers
    pasado = L.Input((VENTANA_H, n_pasado), name="pasado")
    futuro = L.Input((HORIZONTE_H, n_futuro), name="futuro")
    h = L.LSTM(unidades)(pasado)
    z = L.Concatenate()([h, L.Flatten()(futuro)])
    z = L.Dense(256, activation="relu")(z)
    z = L.Dropout(0.1)(z)
    salida = L.Dense(HORIZONTE_H, name="curva")(z)
    red = tf.keras.Model([pasado, futuro], salida)
    red.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    return red


class PronosticadorGas(mlflow.pyfunc.PythonModel):
    """tipo = "lstm" (usa la red) o "ingenuo" (repite la semana anterior)."""

    def load_context(self, context):
        with open(context.artifacts["parametros"], encoding="utf-8") as f:
            self.p = json.load(f)
        self.red = None
        if self.p["tipo"] == "lstm":
            import tensorflow as tf
            self.red = tf.keras.models.load_model(context.artifacts["red"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        salidas = []
        for estacion, g in model_input.groupby("estacion"):
            g = g.sort_values("datetime").tail(VENTANA_H + HORIZONTE_H).reset_index(drop=True)
            if len(g) < VENTANA_H + HORIZONTE_H:
                raise ValueError(f"{estacion}: se necesitan {VENTANA_H + HORIZONTE_H} horas, llegaron {len(g)}")
            futuras = g["datetime"].iloc[VENTANA_H:].to_numpy()

            if self.p["tipo"] == "ingenuo":
                curva = g["valor_corregido"].iloc[:VENTANA_H].to_numpy(float)
            else:
                g = g.copy()
                g["valor_corregido"] = g["valor_corregido"].fillna(0.0)
                valor, cal, onehot = matrices_estacion(g, estacion, self.p["stats"], self.p["estaciones"])
                pasado, futuro = construir_muestra(valor, cal, onehot, VENTANA_H)
                z = self.red.predict([pasado[None], futuro[None]], verbose=0)[0]
                st = self.p["stats"][estacion]
                curva = np.clip(z * st["desv"] + st["media"], 0, None)

            salidas.append(pd.DataFrame({"estacion": estacion, "datetime": futuras, "pronostico": curva}))
        return pd.concat(salidas, ignore_index=True)
