# models/train_models.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

#from models.train_models import train_random_forest
#from models.train_models import train_extra_trees
#from models.train_models import train_xgboost

# ======================================================
# RANDOM FOREST
# ======================================================

def train_random_forest(
    X_train,
    y_train,
    random_state=42
):
    """
    Entrena un Random Forest.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# ======================================================
# EXTRA TREES
# ======================================================

def train_extra_trees(
    X_train,
    y_train,
    random_state=42
):
    """
    Entrena un Extra Trees.
    """

    model = ExtraTreesClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# ======================================================
# XGBOOST
# ======================================================

def train_xgboost(
    X_train,
    y_train,
    random_state=42
):
    """
    Entrena un XGBoost.
    """

    model = XGBClassifier(
        random_state=random_state,
        n_estimators=100,#200
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8, # subsample y colsample_bytree reducen la cant de datos y características utilizadas por árbol, acelerando el entrenamiento
        max_depth=4,#6 genera árboles más pequeños
        eval_metric="mlogloss",
        tree_method='hist', # reduce mucho el uso de memoria
        verbosity=0,
        n_jobs=2 # evita usar todos los núcleos.
    )

    model.fit(X_train, y_train)

    return model

#import joblib
#from pathlib import Path

#def guardar_modelo(modelo, ruta):
#    """
#    Guarda un modelo entrenado.
#    """

#    ruta = Path(ruta)
#    ruta.parent.mkdir(parents=True, exist_ok=True)

#    joblib.dump(modelo, ruta)

#    print(f"Modelo guardado en:\n{ruta}")

