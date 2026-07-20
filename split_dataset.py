import os
import shutil
from sklearn.model_selection import train_test_split
from config import (
    OUTPUT_BASE,
    SPLIT_OUTPUT,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    IMAGE_EXTENSIONS
)

# Crear carpetas de salida
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(SPLIT_OUTPUT, f'dataset_{split}'), exist_ok=True)

# Recorrer cada clase principal
for class_name in os.listdir(OUTPUT_BASE):
    class_path = os.path.join(OUTPUT_BASE, class_name)
    if not os.path.isdir(class_path):
        continue

    # Ver si tiene subcarpetas (subclases)
    subdirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]

    if subdirs:
        # Caso con subclases: procesar cada subclase por separado
        for sub in subdirs:
            sub_path = os.path.join(class_path, sub)
            images = [f for f in os.listdir(sub_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
            if not images:
                continue

            # División estratificada (aunque aquí todas son de la misma subclase)
            train_imgs, temp = train_test_split(images, test_size=VAL_RATIO+TEST_RATIO, random_state=RANDOM_SEED)
            val_imgs, test_imgs = train_test_split(temp, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=RANDOM_SEED)

            # Copiar a cada carpeta de destino
            for split, img_list in zip(['train','val','test'], [train_imgs, val_imgs, test_imgs]):
                dst_dir = os.path.join(SPLIT_OUTPUT, f'dataset_{split}', class_name, sub)
                os.makedirs(dst_dir, exist_ok=True)
                for img in img_list:
                    shutil.copy2(os.path.join(sub_path, img), os.path.join(dst_dir, img))
    else:
        # Caso sin subcarpetas (clase plana)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
        if not images:
            continue

        train_imgs, temp = train_test_split(images, test_size=VAL_RATIO+TEST_RATIO, random_state=RANDOM_SEED)
        val_imgs, test_imgs = train_test_split(temp, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=RANDOM_SEED)

        for split, img_list in zip(['train','val','test'], [train_imgs, val_imgs, test_imgs]):
            dst_dir = os.path.join(SPLIT_OUTPUT, f'dataset_{split}', class_name)
            os.makedirs(dst_dir, exist_ok=True)
            for img in img_list:
                shutil.copy2(os.path.join(class_path, img), os.path.join(dst_dir, img))

print("✅ Partición completada. Revisa las carpetas:")
print(f"   - {SPLIT_OUTPUT}/dataset_train")
print(f"   - {SPLIT_OUTPUT}/dataset_val")
print(f"   - {SPLIT_OUTPUT}/dataset_test")