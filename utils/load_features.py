import numpy as np
import joblib
from pathlib import Path

from config import FEATURES_OUTPUT


def cargar_features(descriptor):
    """
    Carga las características previamente extraídas.

    Parámetros
    ----------
    descriptor : str

        baseline
        hog
        fusion

    Retorna
    -------
    X_train
    X_val
    X_test
    y_train
    y_val
    y_test
    label_encoder
    """
    descriptor = descriptor.lower()

    if descriptor not in ("baseline", "hog", "fusion"):
        raise ValueError(
            f"Descriptor '{descriptor}' no válido."
        )
    
    carpeta = FEATURES_OUTPUT / descriptor

    if not carpeta.exists():
        raise FileNotFoundError(
            f"No existe la carpeta {carpeta}"
        )

    X_train = np.load(carpeta / "X_train.npy")
    X_val   = np.load(carpeta / "X_val.npy")
    X_test  = np.load(carpeta / "X_test.npy")

    y_train = np.load(carpeta / "y_train.npy")
    y_val   = np.load(carpeta / "y_val.npy")
    y_test  = np.load(carpeta / "y_test.npy")

    label_encoder = joblib.load(
        carpeta / "label_encoder.pkl"
    )

    print("="*60)
    print(f"Descriptor: {descriptor}")
    print("="*60)

    print("Train :", X_train.shape)
    print("Val   :", X_val.shape)
    print("Test  :", X_test.shape)


    print()
    print("Etiquetas")

    print("Train :", y_train.shape)
    print("Val   :", y_val.shape)
    print("Test  :", y_test.shape)

    print()

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_encoder
    )