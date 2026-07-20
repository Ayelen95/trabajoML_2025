from pathlib import Path

# ========================================
# Rutas
# ========================================
INPUT_BASE = Path("/home/dao/Documentos/proyectosML/ml_tf/images_2026/images")

OUTPUT_BASE = Path("/home/dao/Documentos/proyectosML/ml_tf/images_2026/images_data_augmentation")

# =============================
# SPLIT
# =============================

SPLIT_OUTPUT = Path("/home/dao/Documentos/proyectosML/ml_tf/images_2026/")

# ========================================
# Dataset
# ========================================
TARGET_PER_CLASS = 1000

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".webp",
)

# =====================================================
# PARTICIÓN
# =====================================================

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

# ========================================
# FEATURES
# ========================================

IMAGE_SIZE = (128, 128)

SAVE_FEATURES = True

FEATURES_OUTPUT = Path(
    "/home/dao/Documentos/proyectosML/ml_tf/features_output"
)

# =====================================================
# FEATURE EXTRACTION
# =====================================================

#IMAGE_SIZE = (128, 128)

#FEATURES_OUTPUT = Path(
 #   "/home/dao/Documentos/proyectosML/ml_tf/features_output"
#)

# Descriptor a utilizar
#FEATURE_METHOD = "baseline"

# Opciones futuras:
# baseline
# hog
# fusion

# =====================================================
# REPRODUCIBILIDAD
# =====================================================

RANDOM_SEED = 42

# ========================================
# EXPERIMENTOS
# ========================================

DATASET_TRAIN = SPLIT_OUTPUT / "dataset_train"

DATASET_VAL = SPLIT_OUTPUT / "dataset_val"

DATASET_TEST = SPLIT_OUTPUT / "dataset_test"

IMG_SIZE = (128, 128)

CATEGORIAS = [
    "cardboard",
    "e-waste",
    "medical",
    "metal",
    "paper",
    "plastic",
]

# ========================================
# HOG
# ========================================

HOG_ORIENTATIONS = 9

HOG_PIXELS_PER_CELL = (16, 16)

HOG_CELLS_PER_BLOCK = (2, 2)

HOG_BLOCK_NORM = "L2-Hys"

# ========================================
# LBP
# ========================================

LBP_RADIUS = 3

LBP_POINTS = 24

LBP_BINS = 64

# ========================================
# HSV
# ========================================

HSV_BINS = 32

# ========================================
# GLCM
# ========================================

GLCM_DISTANCES = [1]

GLCM_ANGLES = [0, 45, 90, 135] #GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

GLCM_LEVELS = 256


# ========================================
# FEATURES
# ========================================

FEATURES_OUTPUT = Path(
    "/home/dao/Documentos/proyectosML/ml_tf/features_output"
)

# ========================================
# SALIDA DE MODELOS
# ========================================

MODELS_OUTPUT = Path(
    "/home/dao/Documentos/proyectosML/ml_tf/models_saved"
)

# ========================================
# RESULTADOS
# ========================================

RESULTS_OUTPUT = Path(
    "/home/dao/Documentos/proyectosML/ml_tf/results"
)