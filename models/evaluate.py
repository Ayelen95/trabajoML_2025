"""
==============================================================
evaluate.py

Funciones para evaluar modelos de Machine Learning.

Este módulo calcula todas las métricas utilizadas en la tesis.

Autor: Daiana Ordóñez
==============================================================
"""
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ============================================================
# EVALUACIÓN GENERAL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    class_names=None
):
    """
    Evalúa un modelo utilizando el conjunto de prueba.

    Parámetros
    ----------
    model :
        Modelo entrenado.

    X_test :
        Características del conjunto de prueba.

    y_test :
        Etiquetas reales.

    class_names :
        Lista con el nombre de las clases.

    Retorna
    -------
    dict
        Diccionario con todas las métricas.
    """

    print()
    print("="*60)
    print("Evaluando modelo...")
    print("="*60)

    # --------------------------------------------------------
    # Predicciones
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # F1 Score
    # --------------------------------------------------------

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Matriz de confusión
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Reporte completo
    # --------------------------------------------------------

    if class_names is not None:

        report = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0
        )

    else:

        report = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

    print()
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)

    print()
    print("Confusion Matrix")
    print(cm)

    print()
    print("Classification Report")
    print(report)

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "confusion_matrix": cm,

        "classification_report": report,

        "y_pred": y_pred,

        "y_prob": y_prob

    }

# ============================================================
# GUARDAR MATRIZ DE CONFUSIÓN
# ============================================================

def guardar_matriz_confusion(
    confusion_matrix,
    class_names,
    output_dir,
    model_name,
    accuracy
):
    """
    Guarda la matriz de confusión como imagen PNG.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_names
    )

    disp.plot(
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=False
    )

    plt.title(
        f"{model_name}\nAccuracy = {accuracy:.4f}"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    output_file = output_dir / f"{model_name}_confusion_matrix.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("="*60)
    print("Matriz de confusión guardada")
    print(output_file)
    print("="*60)

# ============================================================
# GUARDAR CLASSIFICATION REPORT
# ============================================================

def guardar_classification_report(
    report,
    output_dir,
    model_name
):
    """
    Guarda el Classification Report en un archivo TXT.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = output_dir / f"{model_name}_classification_report.txt"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    print()
    print("="*60)
    print("Classification Report guardado")
    print(output_file)
    print("="*60)