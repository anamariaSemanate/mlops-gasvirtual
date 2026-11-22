"""Carga, control de calidad y partición de los datos (Paso 6 del tutorial).

Parte de data/raw/series_anonimizadas.parquet (generado por scripts/anonimizar_datos.py).
"""
import pandas as pd

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, TEST_DIAS

COLUMNAS = ["estacion", "datetime", "valor_corregido",
            "dia_semana", "mes", "es_festivo", "es_fin_de_semana"]


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {RAW_DATA_PATH}. Genera los datos con scripts/anonimizar_datos.py"
        )
    df = pd.read_parquet(RAW_DATA_PATH)
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {faltan}. Encontradas: {list(df.columns)}")
    return df[COLUMNAS]


def control_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Malla horaria completa por estación, sin negativos y con huecos cortos interpolados."""
    partes = []
    for estacion, g in df.groupby("estacion"):
        g = g.drop_duplicates("datetime").set_index("datetime").sort_index()
        malla = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(malla)
        g.index.name = "datetime"
        g["estacion"] = estacion

        g.loc[g["valor_corregido"] < 0, "valor_corregido"] = pd.NA
        g["imputado"] = g["valor_corregido"].isna().astype("int8")
        g["valor_corregido"] = g["valor_corregido"].interpolate(limit=6).ffill().bfill()

        # calendario de las horas que no existían: se reconstruye con la fecha desplazada
        g["dia_semana"] = g["dia_semana"].fillna(pd.Series(g.index.dayofweek, index=g.index))
        g["mes"] = g["mes"].ffill().bfill()
        g["es_fin_de_semana"] = g["es_fin_de_semana"].fillna((g["dia_semana"] >= 5).astype(int))
        g["es_festivo"] = g["es_festivo"].fillna(0)
        partes.append(g.reset_index())

    out = pd.concat(partes, ignore_index=True)
    for c in ["dia_semana", "mes", "es_festivo", "es_fin_de_semana"]:
        out[c] = out[c].astype("int8")
    return out.sort_values(["estacion", "datetime"]).reset_index(drop=True)


def fecha_corte_test(df: pd.DataFrame) -> pd.Timestamp:
    """Inicio del periodo de test: los últimos TEST_DIAS días comunes a todas las estaciones."""
    fin_comun = df.groupby("estacion")["datetime"].max().min()
    return (fin_comun - pd.Timedelta(days=TEST_DIAS)).normalize() + pd.Timedelta(hours=0)


def preprocess_data(guardar: bool = True) -> pd.DataFrame:
    df = control_calidad(load_raw_data())
    if guardar:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_DATA_PATH, index=False)
    return df


def load_processed_data() -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        return pd.read_parquet(PROCESSED_DATA_PATH)
    return preprocess_data()


if __name__ == "__main__":
    data = preprocess_data()
    print(data.head())
    print(data.groupby("estacion").agg(
        desde=("datetime", "min"), hasta=("datetime", "max"),
        horas=("datetime", "size"), imputadas=("imputado", "sum"),
        consumo_medio=("valor_corregido", "mean")))
    print("Inicio del test:", fecha_corte_test(data))
