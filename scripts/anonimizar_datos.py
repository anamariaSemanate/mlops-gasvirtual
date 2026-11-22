"""Anonimiza las series horarias de consumo de GasVirtual antes de publicarlas.

Se ejecuta en el computador de la autora, sobre los datos reales (carpeta datos_base/
del repositorio privado del trabajo de grado). Genera:

  data/raw/series_anonimizadas.parquet   -> se puede subir al repositorio de la tarea
  clave_anonimizacion.json               -> NUNCA se sube (está en .gitignore); permite revertir

Tratamiento aplicado
--------------------
1. Estaciones: los nombres reales se reemplazan por códigos EST_01, EST_02, ...
   asignados en orden aleatorio. Los nombres se leen de los archivos *_base.parquet
   de la carpeta de origen; no quedan escritos en este código. Si dos estaciones
   abastecen el mismo municipio, se combinan antes según estaciones_combinar.json
   (archivo local, fuera del repositorio), por ejemplo:
       {"Municipio": ["Estacion_A", "Estacion_B"]}
2. Volumen: cada serie se multiplica por un factor secreto distinto por estación
   (entre 0.4 y 2.5). Así no se conocen los volúmenes reales, pero las métricas
   relativas (WAPE, sesgo %, MAPE) y la forma de la curva quedan idénticas.
3. Fechas: se desplazan hacia atrás un número secreto de semanas completas (múltiplo
   de 52 semanas = 364 días), de modo que cada hora conserva su día de la semana.
   Las variables de calendario (festivos, fin de semana, mes) se calculan con las
   fechas reales ANTES del desplazamiento y se conservan como columnas.
4. Se eliminan temperatura y presión de medición (el modelo final no las usa y
   la presión de operación identifica la estación) y los nombres de festivos
   (junto con la fecha desplazada permitirían deducir el desplazamiento).

Uso:
    python scripts/anonimizar_datos.py --origen "RUTA/08-Notebooks/datos_base"
"""
import argparse
import json
import secrets
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "data" / "raw" / "series_anonimizadas.parquet"
CLAVE = RAIZ / "clave_anonimizacion.json"
COMBINAR = RAIZ / "estaciones_combinar.json"
CALENDARIO = ["dia_semana", "mes", "es_festivo", "es_fin_de_semana"]
COLUMNAS_SALIDA = ["estacion", "datetime", "valor_corregido"] + CALENDARIO


def cargar(origen: Path, estacion: str) -> pd.DataFrame:
    df = pd.read_parquet(origen / f"{estacion}_base.parquet")
    return df.sort_values("datetime").reset_index(drop=True)


def combinar(origen: Path, nombre: str, estaciones: list[str]) -> pd.DataFrame:
    """Suma el volumen hora a hora de estaciones que abastecen el mismo municipio."""
    df = None
    for est in estaciones:
        g = cargar(origen, est)[["datetime", "valor_corregido"] + CALENDARIO]
        if df is None:
            df = g
            continue
        df = df.merge(g, on="datetime", how="outer", suffixes=("", "_otro"))
        df["valor_corregido"] = df[["valor_corregido", "valor_corregido_otro"]].sum(axis=1, min_count=1)
        for col in CALENDARIO:
            df[col] = df[col].combine_first(df[f"{col}_otro"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_otro")])
    df["estacion"] = nombre
    return df.sort_values("datetime").reset_index(drop=True)


def cargar_series_reales(origen: Path) -> pd.DataFrame:
    grupos = json.loads(COMBINAR.read_text(encoding="utf-8")) if COMBINAR.exists() else {}
    combinadas = {e for lista in grupos.values() for e in lista}
    todas = sorted(f.name.removesuffix("_base.parquet") for f in origen.glob("*_base.parquet"))
    if not todas:
        raise FileNotFoundError(f"No hay archivos *_base.parquet en {origen}")
    partes = [cargar(origen, e).assign(estacion=e.replace("_", " ")) for e in todas if e not in combinadas]
    partes += [combinar(origen, nombre, lista) for nombre, lista in grupos.items()]
    return pd.concat([p[COLUMNAS_SALIDA] for p in partes], ignore_index=True)


def generar_clave(estaciones: list[str]) -> dict:
    rng = np.random.default_rng(secrets.randbits(64))
    orden = rng.permutation(len(estaciones))
    codigos = {est: f"EST_{i + 1:02d}" for i, est in zip(orden, estaciones)}
    factores = {est: round(float(rng.uniform(0.4, 2.5)), 4) for est in estaciones}
    semanas = int(52 * rng.integers(2, 6))  # entre 2 y 5 años de 364 días
    return {"codigos": codigos, "factores": factores, "desplazamiento_semanas": semanas}


def anonimizar(real: pd.DataFrame, clave: dict) -> pd.DataFrame:
    df = real.copy()
    df["valor_corregido"] = (df["valor_corregido"] * df["estacion"].map(clave["factores"])).round(3)
    df["datetime"] = df["datetime"] - pd.Timedelta(weeks=clave["desplazamiento_semanas"])
    df["estacion"] = df["estacion"].map(clave["codigos"])
    for col in CALENDARIO:
        df[col] = df[col].astype("int8")
    return df.sort_values(["estacion", "datetime"]).reset_index(drop=True)[COLUMNAS_SALIDA]


def revertir(anon: pd.DataFrame, clave: dict) -> pd.DataFrame:
    """Recupera estación, fecha y volumen reales (útil para validar resultados en privado)."""
    inverso = {v: k for k, v in clave["codigos"].items()}
    df = anon.copy()
    df["estacion"] = df["estacion"].map(inverso)
    df["valor_corregido"] = df["valor_corregido"] / df["estacion"].map(clave["factores"])
    df["datetime"] = df["datetime"] + pd.Timedelta(weeks=clave["desplazamiento_semanas"])
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--origen", required=True, type=Path, help="carpeta datos_base/ con los parquet reales")
    p.add_argument("--nueva-clave", action="store_true", help="genera una clave nueva aunque ya exista una")
    args = p.parse_args()

    real = cargar_series_reales(args.origen)
    estaciones = sorted(real["estacion"].unique())

    if CLAVE.exists() and not args.nueva_clave:
        clave = json.loads(CLAVE.read_text(encoding="utf-8"))
        print(f"Usando la clave existente: {CLAVE.name}")
    else:
        clave = generar_clave(estaciones)
        CLAVE.write_text(json.dumps(clave, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Clave nueva guardada en {CLAVE.name} (no la subas al repositorio)")

    anon = anonimizar(real, clave)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    anon.to_parquet(SALIDA, index=False)

    # Comprobación: la reversión recupera exactamente los datos reales (salvo redondeo)
    rec = revertir(anon, clave).sort_values(["estacion", "datetime"]).reset_index(drop=True)
    ref = real.sort_values(["estacion", "datetime"]).reset_index(drop=True)
    err = np.nanmax(np.abs(rec["valor_corregido"].to_numpy() - ref["valor_corregido"].to_numpy()))
    assert (rec["datetime"].to_numpy() == ref["datetime"].to_numpy()).all()
    print(f"Filas: {len(anon):,} | estaciones: {anon['estacion'].nunique()} | "
          f"rango: {anon['datetime'].min():%Y-%m-%d} a {anon['datetime'].max():%Y-%m-%d} | "
          f"error máx. de reversión: {err:.4f} m³")
    print(f"Archivo anonimizado: {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
