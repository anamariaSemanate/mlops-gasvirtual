"""Entrenamiento y registro de experimentos en MLflow (Paso 9 del tutorial).

Entrena:
  - ingenuo estacional (repite la semana anterior): línea base
  - LSTM global con salida directa de 168 h, con 3 semillas

Evalúa cada modelo con pronósticos que arrancan cada lunes del periodo de test
(walk-forward) y registra en MLflow las métricas, gráficas y el modelo empaquetado.

Ejecuta:  python -m src.train
"""
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.config import (EPOCAS, EXPERIMENT_NAME, FIGURES_DIR, METRICS_DIR, MLFLOW_TRACKING_URI,
                        PASO_MUESTRAS_H, ROOT_DIR, SEMILLAS, VAL_FRACCION, VENTANA_H, HORIZONTE_H)
from src.data import fecha_corte_test, load_processed_data
from src.features import crear_ventanas, estadisticas_normalizacion
from src.metrics import metricas_semana, resumir
from src.modelo import PronosticadorGas, construir_red


def desnormalizar(Z: np.ndarray, meta: pd.DataFrame, stats: dict) -> np.ndarray:
    media = meta["estacion"].map(lambda e: stats[e]["media"]).to_numpy()[:, None]
    desv = meta["estacion"].map(lambda e: stats[e]["desv"]).to_numpy()[:, None]
    return Z * desv + media


def evaluar(pred: np.ndarray, real: np.ndarray, meta: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    filas = []
    for k, (estacion, inicio) in enumerate(meta.itertuples(index=False)):
        m = metricas_semana(real[k], pred[k])
        filas.append({"estacion": estacion, "inicio": inicio, "total_real": real[k].sum(), **m})
    detalle = pd.DataFrame(filas)
    metricas = resumir(filas)
    for estacion, g in detalle.groupby("estacion"):
        r = resumir(g.to_dict("records"))
        metricas[f"wape_diario_{estacion}"] = r["wape_diario"]
        metricas[f"sesgo_{estacion}"] = r["sesgo"]
    return metricas, detalle


def grafica_semanas(pred, real, meta, nombre) -> Path:
    """Última semana de test de cada estación: real vs. pronóstico."""
    ult = meta.groupby("estacion").tail(1).index
    fig, axes = plt.subplots(len(ult), 1, figsize=(11, 2.3 * len(ult)), sharex=True)
    for ax, k in zip(np.atleast_1d(axes), ult):
        ax.plot(real[k], label="real", color="#444")
        ax.plot(pred[k], label="pronóstico", color="#7b3fe4")
        ax.set_title(f"{meta.loc[k, 'estacion']} · semana desde {meta.loc[k, 'inicio']:%Y-%m-%d}", fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("hora del horizonte")
    fig.tight_layout()
    ruta = FIGURES_DIR / f"semanas_{nombre}.png"
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    return ruta


def registrar_run(nombre, params, pred, real, meta, parametros_modelo, red=None) -> dict:
    with mlflow.start_run(run_name=nombre) as run:
        mlflow.log_params(params)
        metricas, detalle = evaluar(pred, real, meta)
        mlflow.log_metrics(metricas)

        ruta_det = METRICS_DIR / f"detalle_semanas_{nombre}.csv"
        detalle.to_csv(ruta_det, index=False)
        mlflow.log_artifact(str(ruta_det), artifact_path="metrics")
        mlflow.log_artifact(str(grafica_semanas(pred, real, meta, nombre)), artifact_path="figures")

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = {"parametros": str(Path(tmp) / "parametros.json")}
            Path(artifacts["parametros"]).write_text(json.dumps(parametros_modelo), encoding="utf-8")
            if red is not None:
                artifacts["red"] = str(Path(tmp) / "red.keras")
                red.save(artifacts["red"])
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=PronosticadorGas(),
                artifacts=artifacts,
                code_paths=[str(ROOT_DIR / "src")],
                pip_requirements=["mlflow", "pandas", "numpy", "tensorflow"],
            )
        mlflow.set_tag("model_uri", info.model_uri)
        print(f"{nombre:<14} WAPE diario {metricas['wape_diario']:.3%} | sesgo {metricas['sesgo']:+.2%} "
              f"| semanas subestimadas {metricas['pct_semanas_subestimadas']:.0%}")
        return {"modelo": nombre, **params, **metricas, "run_id": run.info.run_id}


def train_models():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_processed_data()
    corte = fecha_corte_test(df)
    estaciones = sorted(df["estacion"].unique())
    stats = estadisticas_normalizacion(df[df["datetime"] < corte])
    base_modelo = {"stats": stats, "estaciones": estaciones}

    # entrenamiento: ventanas cuyo horizonte completo termina antes del test
    train_df = df[df["datetime"] < corte]
    Xp, Xf, Y, meta_tr = crear_ventanas(train_df, stats, estaciones, paso_h=PASO_MUESTRAS_H)
    corte_val = meta_tr["inicio"].quantile(1 - VAL_FRACCION)
    es_val = (meta_tr["inicio"] >= corte_val).to_numpy()
    # test: un pronóstico por estación cada lunes desde el corte (la historia puede venir de antes)
    Tp, Tf, TY, meta_te = crear_ventanas(df, stats, estaciones, desde=corte, solo_lunes=True)
    real = desnormalizar(TY, meta_te, stats)
    print(f"Corte de test: {corte:%Y-%m-%d} | ventanas train {(~es_val).sum()} · val {es_val.sum()} "
          f"· test {len(meta_te)} ({meta_te['inicio'].nunique()} lunes × {len(estaciones)} estaciones)")

    resultados = []
    comunes = {"ventana_h": VENTANA_H, "horizonte_h": HORIZONTE_H, "test_desde": str(corte.date())}

    # 1) ingenuo estacional: el pronóstico es la semana anterior
    pred = desnormalizar(Tp[:, :, 0], meta_te, stats)
    resultados.append(registrar_run("ingenuo", {**comunes, "tipo": "ingenuo", "semilla": 0},
                                    pred, real, meta_te, {**base_modelo, "tipo": "ingenuo"}))

    # 2) LSTM global con varias semillas
    import tensorflow as tf
    for semilla in SEMILLAS:
        red = construir_red(Xp.shape[2], Xf.shape[2], semilla)
        hist = red.fit(
            [Xp[~es_val], Xf[~es_val]], Y[~es_val],
            validation_data=([Xp[es_val], Xf[es_val]], Y[es_val]),
            epochs=EPOCAS, batch_size=64, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )
        pred = desnormalizar(red.predict([Tp, Tf], verbose=0), meta_te, stats)
        pred = np.clip(pred, 0, None)
        params = {**comunes, "tipo": "lstm", "semilla": semilla, "unidades": 64,
                  "epocas_max": EPOCAS, "epocas_usadas": len(hist.history["loss"])}
        resultados.append(registrar_run(f"lstm_s{semilla}", params, pred, real, meta_te,
                                        {**base_modelo, "tipo": "lstm"}, red=red))

    tabla = pd.DataFrame(resultados)
    tabla.to_csv(METRICS_DIR / "comparacion_modelos.csv", index=False)
    print(f"\nTabla comparativa: {METRICS_DIR / 'comparacion_modelos.csv'}")


if __name__ == "__main__":
    train_models()
