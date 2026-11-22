"""SCADA simulado: recorre las semanas de test como si el sistema estuviera en operación.

Cada "lunes" pide a la API el pronóstico de cada estación y, como en la simulación
el consumo real ya se conoce, llama a /evaluacion para que Prometheus/Grafana
muestren cómo evoluciona el error, el sesgo y la deriva semana a semana.

Ejecuta (con la API corriendo):  python scripts/simular_operacion.py --pausa 10
"""
import argparse
import time

import requests

p = argparse.ArgumentParser()
p.add_argument("--api", default="http://127.0.0.1:8000")
p.add_argument("--pausa", type=float, default=10, help="segundos entre semanas (Prometheus lee cada 5 s)")
p.add_argument("--repeticiones", type=int, default=1, help="veces que se recorre el periodo de test")
args = p.parse_args()

op = requests.get(f"{args.api}/estaciones", timeout=30).json()
print(f"{len(op['lunes_disponibles'])} semanas × {len(op['estaciones'])} estaciones")

for _ in range(args.repeticiones):
    for lunes in op["lunes_disponibles"]:
        linea = [lunes[:10]]
        for est in op["estaciones"]:
            r = requests.post(f"{args.api}/evaluacion", json={"estacion": est, "inicio": lunes}, timeout=60)
            if r.ok:
                m = r.json()
                linea.append(f"{est}: WAPE {m['wape_diario']:.1%} sesgo {m['sesgo']:+.1%}")
            else:
                linea.append(f"{est}: error {r.status_code}")
        print(" | ".join(linea))
        time.sleep(args.pausa)
