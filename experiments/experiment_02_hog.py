import sys
from pathlib import Path

# Agregar la carpeta raíz del proyecto al PATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
import joblib

from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from config import (
    DATASET_TRAIN,
    DATASET_VAL,
    DATASET_TEST,
    IMG_SIZE,
    CATEGORIAS,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    HOG_BLOCK_NORM,
    FEATURES_OUTPUT,
)

# ==========================================================
# CARGA DEL DATASET
# ==========================================================

def cargar_imagenes(ruta_base, img_size=IMG_SIZE):
    X = []
    y = []
    for categoria in os.listdir(ruta_base):
        ruta_categoria = os.path.join(ruta_base, categoria)
        if not os.path.isdir(ruta_categoria):
            continue
        for root, dirs, files in os.walk(ruta_categoria):
            for archivo in files:
                if archivo.lower().endswith(
                    (".jpg",".jpeg",".png",".bmp",".webp")
                ):
                    ruta_img = os.path.join(root, archivo)
                    try:
                        img = cv2.imread(ruta_img)
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, img_size)
                        img = img.astype(np.uint8)
                        X.append(img)
                        y.append(categoria)
                    except Exception as e:
                        print(f"Error leyendo {ruta_img}")
                        print(e)
    return np.array(X), np.array(y)


print("="*60)
print("Cargando imágenes...")
print("="*60)

X_train, y_train = cargar_imagenes(str(DATASET_TRAIN))
X_val, y_val = cargar_imagenes(str(DATASET_VAL))
X_test, y_test = cargar_imagenes(str(DATASET_TEST))
print()
print("Train :", X_train.shape)
print("Val   :", X_val.shape)
print("Test  :", X_test.shape)



# ==========================================================
# CODIFICACIÓN DE ETIQUETAS
# ==========================================================

print()
print("="*60)
print("Codificando etiquetas")
print("="*60)
le = LabelEncoder()
le.fit(CATEGORIAS)
y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

print()

print("Mapa de clases")

for clase, codigo in zip(le.classes_, le.transform(le.classes_)):

    print(f"{codigo} -> {clase}")


# ==========================================================
# HOG
# ==========================================================

def extraer_hog(imagenes):
    """
    Extrae descriptores HOG de un conjunto de imágenes.

    Proceso:
    Imagen RGB
        ↓
    Escala de grises
        ↓
    HOG
        ↓
    Vector de características
    """

    features = []

    for img in imagenes:

        # HOG trabaja en escala de grises
        img_gray = rgb2gray(img)

        vector = hog(
            img_gray,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm=HOG_BLOCK_NORM,
            visualize=False
        )

        features.append(vector)

    return np.array(features)


print()

print("="*60)
print("Extrayendo HOG...")
print("="*60)

X_train_hog = extraer_hog(X_train)

X_val_hog = extraer_hog(X_val)

X_test_hog = extraer_hog(X_test)

print()

print("Forma entrenamiento :", X_train_hog.shape)

print("Forma validación    :", X_val_hog.shape)

print("Forma test          :", X_test_hog.shape)

print()

print("HOG listo.")


def guardar_features(
    output_dir,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    label_encoder,
):
    """
    Guarda todas las características de un experimento.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "X_test.npy", X_test)

    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "y_test.npy", y_test)

    joblib.dump(
        label_encoder,
        output_dir / "label_encoder.pkl"
    )

    print()
    print("Características guardadas correctamente.")

from utils.save_features import guardar_features

guardar_features(
    FEATURES_OUTPUT / "hog",
    X_train_hog,
    X_val_hog,
    X_test_hog,
    y_train_enc,
    y_val_enc,
    y_test_enc,
    le
)