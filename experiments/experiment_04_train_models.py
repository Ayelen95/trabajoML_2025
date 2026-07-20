"""
==============================================================
EXPERIMENTO 04

Entrenamiento y evaluación de modelos clásicos
utilizando los descriptores previamente extraídos.

Modelos:
    • Random Forest
    • Extra Trees
    • XGBoost

Autor:
    Daiana Ordoñez - Daniela Aguilar
==============================================================
"""
# python experiments/experiment_04_train_models.py --feature_set baseline
import sys
from pathlib import Path
import gc

# ==========================================================
# Agregar carpeta raíz al PATH
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# ==========================================================
# Imports del proyecto
# ==========================================================
# ==========================================================
# CONFIG
# ==========================================================
from config import RESULTS_OUTPUT
from models.evaluate import (
    evaluate_model,
    guardar_matriz_confusion,
    guardar_classification_report
)

from models.train_models import (
    train_random_forest,
    train_extra_trees,
    train_xgboost,
)

from utils.load_features import cargar_features
from utils.save_results import guardar_resultados
from utils.save_model import save_model
# ==========================================================
# CONFIGURACIÓN DESDE LA LÍNEA DE COMANDOS
# ==========================================================

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--feature_set",
    default="baseline",
    choices=["baseline", "hog", "fusion"],
    help="Descriptor de características"
)

args = parser.parse_args()

FEATURE_SET = args.feature_set

# ==========================================================
# CARGAR FEATURES
# ==========================================================

print("="*60)
print("Cargando características...")
print("="*60)

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    le
) = cargar_features(FEATURE_SET)

# ==========================================================
# MODELOS A ENTRENAR
# ==========================================================

modelos = {

    "RandomForest":
        train_random_forest,

    "ExtraTrees":
        train_extra_trees,

    "XGBoost":
        train_xgboost

}

# ==========================================================
# ENTRENAMIENTO
# ==========================================================

for nombre_modelo, funcion_entrenamiento in modelos.items():

    print()
    print("="*60)
    print(f"Entrenando {nombre_modelo}")
    print("="*60)

    modelo = funcion_entrenamiento(
        X_train,
        y_train
    )

    # ======================================================
    # EVALUACIÓN
    # ======================================================

    resultados = evaluate_model(
        modelo,
        X_test,
        y_test,
        class_names=le.classes_
    )

    # ======================================================
    # GUARDAR RESULTADOS CSV
    # ======================================================

    guardar_resultados(
        nombre_experimento=FEATURE_SET,
        nombre_modelo=nombre_modelo,
        accuracy=resultados["accuracy"],
        precision=resultados["precision"],
        recall=resultados["recall"],
        f1=resultados["f1_score"],
        ruta_salida=RESULTS_OUTPUT
    )

    # ==========================================
    # Guardar matriz de confusión
    # ==========================================

    guardar_matriz_confusion(
        confusion_matrix=resultados["confusion_matrix"],
        class_names=le.classes_,
        output_dir=RESULTS_OUTPUT / FEATURE_SET,
        model_name=nombre_modelo,
        accuracy=resultados["accuracy"]
    )

    # ==========================================
    # Guardar Classification Report
    # ==========================================

    guardar_classification_report(
        report=resultados["classification_report"],
        output_dir=RESULTS_OUTPUT / FEATURE_SET,
        model_name=nombre_modelo
    )

    # ======================================================
    # GUARDAR MODELO ENTRENADO
    # ======================================================

    ruta_modelo = (

        ROOT_DIR
        / "models_saved"
        / FEATURE_SET
        / f"{nombre_modelo}.pkl"
    )

    save_model(
        model=modelo,
        output_file=ruta_modelo
    )

    # ==========================================
    # Liberar memoria
    # ==========================================

    del modelo
    gc.collect()

print()
print("="*60)
print("Experimento finalizado correctamente")
print("="*60)
