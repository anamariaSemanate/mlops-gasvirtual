"""Tablero del operador (Paso 13 del tutorial).

Ejecuta:  streamlit run app/streamlit_app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="GasVirtual · Planeación de despachos", page_icon="⛽", layout="wide")
st.title("Planeación de despachos de gas virtual")
st.caption("Pronóstico de consumo a 7 días y autonomía por estación. "
           "Datos anonimizados; la operación se simula con el histórico (sin conexión al SCADA).")


@st.cache_data(ttl=300)
def opciones():
    r = requests.get(f"{API_URL}/estaciones", timeout=30)
    r.raise_for_status()
    return r.json()


try:
    op = opciones()
except requests.exceptions.RequestException:
    st.error("No se pudo conectar con la API. Asegúrate de que FastAPI esté corriendo en " + API_URL)
    st.stop()

with st.sidebar:
    st.header("Parámetros")
    estacion = st.selectbox("Estación", op["estaciones"])
    lunes = [pd.Timestamp(t) for t in op["lunes_disponibles"]]
    inicio = st.selectbox("Semana a planear (desde)", lunes, index=len(lunes) - 1,
                          format_func=lambda t: t.strftime("%Y-%m-%d (lunes)"))
    inventario_dias = st.slider("Inventario actual (días de consumo promedio)", 0.5, 7.0, 3.0, 0.1)
    reserva_dias = st.slider("Reserva de seguridad (días)", 0.0, 2.0, 0.8, 0.1)
    tiempo_viaje_h = st.number_input("Tiempo de viaje del camión (h)", 1, 48, 6)
    calcular = st.button("Calcular", type="primary", use_container_width=True)

if calcular:
    payload = {"estacion": estacion, "inicio": inicio.isoformat(), "inventario_dias": inventario_dias,
               "reserva_dias": reserva_dias, "tiempo_viaje_h": tiempo_viaje_h}
    try:
        aut = requests.post(f"{API_URL}/autonomia", json=payload, timeout=60)
        pro = requests.post(f"{API_URL}/pronostico", json=payload, timeout=60)
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con la API.")
        st.stop()
    if not aut.ok:
        st.error(aut.json().get("detail", aut.text))
        st.stop()
    a, p = aut.json(), pro.json()

    dias = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
    fmt = lambda t: f"{dias[pd.Timestamp(t).dayofweek]} {pd.Timestamp(t):%d/%m %H:%M}" if t else "—"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Autonomía", f"{a['autonomia_horas']:.0f} h" if a["autonomia_horas"] is not None else "> 7 días")
    c2.metric("Llega a la reserva", fmt(a["hora_reserva"]))
    c3.metric("Despacho sugerido", fmt(a["despacho_sugerido"]))
    c4.metric("Consumo pronosticado 7 días", f"{p['consumo_7d']:,.0f}")

    if a["alcanza_horizonte"]:
        st.success("El inventario alcanza para toda la semana sin tocar la reserva.")
    else:
        st.warning(f"Programar despacho antes de {fmt(a['despacho_sugerido'])}.")

    curva = pd.DataFrame(p["curva"]).set_index("datetime")
    curva.index = pd.to_datetime(curva.index)
    st.subheader("Consumo por hora")
    st.line_chart(curva, color=["#7b3fe4", "#444444"][: curva.shape[1]])

    inv = pd.DataFrame({"inventario restante": a["inventario_restante"],
                        "reserva de seguridad": a["reserva_seguridad"]}, index=curva.index)
    st.subheader("Inventario proyectado")
    st.line_chart(inv, color=["#1f77b4", "#d62728"])

    if "real" in curva:
        total_r = curva["real"].sum()
        sesgo = (curva["pronostico"].sum() - total_r) / total_r
        st.info(f"Simulación: el consumo real de esa semana fue {total_r:,.0f} "
                f"(el pronóstico se desvió {sesgo:+.1%}).")
else:
    st.info("Elige la estación, la semana y el inventario, y presiona **Calcular**.")
