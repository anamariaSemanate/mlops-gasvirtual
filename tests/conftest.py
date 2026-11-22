import os

# Las pruebas no necesitan el servidor de MLflow ni el modelo entrenado
os.environ.setdefault("CARGAR_MODELO", "0")
