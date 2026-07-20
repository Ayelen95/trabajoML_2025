import subprocess
import sys


def ejecutar(script, argumentos=None):

    comando = [sys.executable, script]

    if argumentos:
        comando.extend(argumentos)

    print()
    print("=" * 70)
    print("Ejecutando:")
    print(" ".join(comando))
    print("=" * 70)

    subprocess.run(
        comando,
        check=True
    )


# =====================================================
# BASELINE
# =====================================================

ejecutar(
    "experiments/experiment_01_baseline.py"
)

ejecutar(
    "experiments/experiment_04_train_models.py",
    ["--feature_set", "baseline"]
)


# =====================================================
# HOG
# =====================================================

ejecutar(
    "experiments/experiment_02_hog.py"
)

ejecutar(
    "experiments/experiment_04_train_models.py",
    ["--feature_set", "hog"]
)


# =====================================================
# FUSIÓN
# =====================================================

ejecutar(
    "experiments/experiment_03_fusion.py"
)

ejecutar(
    "experiments/experiment_04_train_models.py",
    ["--feature_set", "fusion"]
)

print()
print("=" * 70)
print("PIPELINE COMPLETO FINALIZADO")
print("=" * 70)