import sys
from pathlib import Path

# Agregar la carpeta raíz del proyecto al PATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import os
import cv2
import numpy as np
import joblib

from pathlib import Path

from skimage.feature import hog
from skimage.color import rgb2gray

from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

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
    LBP_RADIUS,
    LBP_POINTS,
    LBP_BINS,
    HSV_BINS,
    GLCM_DISTANCES,
    GLCM_ANGLES,
    GLCM_LEVELS,
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

# ==========================================================
# LBP
# ==========================================================

def extraer_lbp(imagenes, radio = LBP_RADIUS, n_points=LBP_POINTS, n_bins=LBP_BINS):
    """
    Extrae descriptores Local Binary Pattern (LBP).
    Proceso

    Imagen RGB
          ↓
    Escala de grises
          ↓
    Local Binary Pattern
          ↓
    Histograma normalizado
          ↓ 
    Vector de características
    """

    features = []

    for img in imagenes:
        img_gray = (rgb2gray(img)*255).astype(np.uint8) if img.ndim == 3 else img

        lbp = local_binary_pattern(img_gray, n_points, radio, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        features.append(hist)

    return np.array(features)


print()

print("="*60)
print("Extrayendo LBP...")
print("="*60)

X_train_lbp = extraer_lbp(X_train)
X_val_lbp = extraer_lbp(X_val)
X_test_lbp = extraer_lbp(X_test)

print()

print("Forma entrenamiento :", X_train_lbp.shape)
print("Forma validación    :", X_val_lbp.shape)
print("Forma test          :", X_test_lbp.shape)
print()
print("LBP listo.")

# ==========================================================
# HSV
# ==========================================================


def extraer_hsv(imagenes, bins=HSV_BINS):
    """
    Extrae histogramas de color en el espacio HSV.
    Proceso

    Imagen RGB
          ↓
    Conversión a HSV
          ↓
    Histograma de H
    Histograma de S
    Histograma de V
          ↓
    Concatenación
          ↓
    Vector de características
    """

    features = []

    for img in imagenes:

        # Convetimos de RGB a HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        canales = []
        for canal in range(3): #H, S, V
            canal_actual = img_hsv[:,:, canal]
            if canal_actual.sum() == 0:
                hist = np.zeros(bins)
            else:
                hist, _ = np.histogram(
                    canal_actual.ravel(), # aplana la imagen
                    bins=bins, 
                    range=(0, 256), 
                    density=True
                )
            canales.append(hist)
        

        features.append(np.concatenate(canales))

    return np.array(features)

print()

print("="*60)
print("Extrayendo HSV...")
print("="*60)

X_train_hsv = extraer_hsv(X_train)
X_val_hsv = extraer_hsv(X_val)
X_test_hsv = extraer_hsv(X_test)

print()

print("Forma entrenamiento :", X_train_hsv.shape)
print("Forma validación    :", X_val_hsv.shape)
print("Forma test          :", X_test_hsv.shape)
print()
print("HSV listo.")

# ==========================================================
# GLCM
# ==========================================================


def extraer_glcm(imagenes, angles=GLCM_ANGLES, distances = GLCM_DISTANCES, levels=GLCM_LEVELS):
    """
    Extrae características de textura mediante
    Gray Level Co-occurrence Matrix (GLCM).
    Proceso
    Imagen RGB
          ↓
    Escala de grises
          ↓
    GLCM
          ↓
    Extracción de propiedades
    • Contraste
    • Disimilitud
    • Homogeneidad
    • Energía
    • Correlación
          ↓
    Vector de características
    """

    features = []

    angles = np.deg2rad(angles)

    for img in imagenes:

        # GLCM requiere escala de grises en formato entero (0-255)
        img_gray = (rgb2gray(img)*255).astype(np.uint8) if img.ndim == 3 else img

        # Calculamos la matriz a distancia 1, en 4 direcciones espaciales 
        glcm = graycomatrix(img_gray, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

        # Extraemos las 5 métricas clave y las aplanamos en un solo vector
        contrast = graycoprops(glcm, 'contrast').ravel()
        dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
        homogeneity = graycoprops(glcm,'homogeneity').ravel()
        energy = graycoprops(glcm, 'energy').ravel()
        correlation = graycoprops(glcm, 'correlation').ravel()

        # Unimos todas las propiedades (5 propiedades * 4 angulos = 20 dimensiones)
        vector_glcm = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])

        features.append(vector_glcm)

    return np.array(features)

print()

print("="*60)
print("Extrayendo GLCM...")
print("="*60)

X_train_glcm = extraer_glcm(X_train)
X_val_glcm = extraer_glcm(X_val)
X_test_glcm = extraer_glcm(X_test)

print()

print("Forma entrenamiento :", X_train_glcm.shape)
print("Forma validación    :", X_val_glcm.shape)
print("Forma test          :", X_test_glcm.shape)
print()
print("GLCM listo.")


print('==================================================================')
print('FUSION')
print('==================================================================')
X_train_fusion = np.concatenate(
    (
        X_train_hog,
        X_train_lbp,
        X_train_hsv,
        X_train_glcm,
    ),
    axis=1,
)

X_val_fusion = np.concatenate(
    (
        X_val_hog,
        X_val_lbp,
        X_val_hsv,
        X_val_glcm,
    ),
    axis=1,
)

X_test_fusion = np.concatenate(
    (
        X_test_hog,
        X_test_lbp,
        X_test_hsv,
        X_test_glcm,
    ),
    axis=1,
)
print("\n" + "=" * 60)
print("RESUMEN DE LA FUSIÓN")
print("=" * 60)

print(f"HOG   : {X_train_hog.shape[1]} características")
print(f"LBP   : {X_train_lbp.shape[1]} características")
print(f"HSV   : {X_train_hsv.shape[1]} características")
print(f"GLCM  : {X_train_glcm.shape[1]} características")
print("-" * 60)
print(f"TOTAL : {X_train_fusion.shape[1]} características")



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
    FEATURES_OUTPUT / "fusion",
    X_train_fusion,
    X_val_fusion,
    X_test_fusion,
    y_train_enc,
    y_val_enc,
    y_test_enc,
    le
)