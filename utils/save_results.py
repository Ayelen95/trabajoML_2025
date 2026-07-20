"""
===========================================================
save_results.py
===========================================================

Guarda los resultados de cada modelo en formato CSV.

Autor: Daiana Ordoñez
Tesis 2026
===========================================================
"""

from pathlib import Path
import pandas as pd
from datetime import datetime


def guardar_resultados(
    nombre_experimento,
    nombre_modelo,
    accuracy,
    precision,
    recall,
    f1,
    ruta_salida
):
    """
    Guarda una fila de resultados.

    Si el archivo existe agrega una nueva fila.

    Si no existe crea el CSV.
    """

    ruta_salida = Path(ruta_salida)

    ruta_salida.mkdir(parents=True, exist_ok=True)

    archivo = ruta_salida / "results.csv"

    fila = pd.DataFrame([{
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Experimento": nombre_experimento,
        "Modelo": nombre_modelo,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4)
    }])

    if archivo.exists():

        viejo = pd.read_csv(archivo)

        nuevo = pd.concat(
            [viejo, fila],
            ignore_index=True
        )

    else:

        nuevo = fila

    nuevo.to_csv(
        archivo,
        index=False
    )

    print()

    print("="*60)
    print("Resultados guardados")
    print(f"Experimento : {nombre_experimento}")
    print(f"Modelo      : {nombre_modelo}")
    print(f"Archivo     : {archivo}")
    print("="*60)