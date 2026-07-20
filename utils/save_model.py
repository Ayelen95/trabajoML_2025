"""
==========================================================
save_model.py

Funciones para guardar modelos entrenados.

Autor: Daiana Ordóñez
==========================================================
"""

from pathlib import Path
import joblib


# ==========================================================
# GUARDAR MODELO
# ==========================================================

def save_model(model, output_file):
    """
    Guarda un modelo entrenado.

    Parámetros
    ----------
    model :
        Modelo entrenado.

    output_file :
        Ruta completa donde se almacenará el modelo.

    Ejemplo
    --------

    save_model(
        model=rf,
        output_file="models_saved/baseline/RandomForest.pkl"
    )
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        output_file
    )

    print()
    print("=" * 60)
    print("Modelo guardado correctamente")
    print("=" * 60)
    print(output_file)