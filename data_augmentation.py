import os
import shutil
import random
import cv2
import numpy as np
import csv

from config import (
    INPUT_BASE,
    OUTPUT_BASE,
    TARGET_PER_CLASS,
    IMAGE_EXTENSIONS
)
from config import RANDOM_SEED
# ===================================================
# FUNCIONES AUXILIARES
# ===================================================
def get_images_from_dir(dir_path):
    """Devuelve lista de rutas completas de imágenes en dir_path (sin recursión)."""
    if not os.path.isdir(dir_path):
        return []
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)]

def get_subclass_structure(class_path):
    """
    Detecta si la clase tiene subcarpetas.
    Retorna: (tipo, lista)
    - tipo = 'flat': lista de imágenes en la raíz de class_path
    - tipo = 'nested': diccionario {subclase: [lista_imágenes]}
    """
    items = os.listdir(class_path)
    subdirs = [item for item in items if os.path.isdir(os.path.join(class_path, item))]
    if subdirs:
        # Estructura anidada
        nested = {}
        for sub in subdirs:
            sub_path = os.path.join(class_path, sub)
            nested[sub] = get_images_from_dir(sub_path)
        return 'nested', nested
    else:
        # Plana
        return 'flat', get_images_from_dir(class_path)

def adjust_brightness(img, delta):
    """Ajusta brillo de forma segura (sin overflow). delta puede ser negativo."""
    # Convertir a int16 para evitar overflow
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + delta, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def copy_and_augment(src_images, dst_dir, target_count):
    """
    src_images: lista de rutas originales.
    dst_dir: carpeta destino (se crea).
    target_count: número final deseado.
    
    Si target_count <= len(src_images): submuestreo (copia aleatoria).
    Si target_count > len(src_images): copia todos + genera aumentadas hasta completar.
    Las aumentadas se nombran con sufijo.
    """
    os.makedirs(dst_dir, exist_ok=True)
    num_orig = len(src_images)
    
    # Submuestreo: elegir aleatoriamente target_count imágenes
    if num_orig >= target_count:
        selected = random.sample(src_images, target_count)
        for src in selected:
            dst = os.path.join(dst_dir, os.path.basename(src))
            shutil.copy2(src, dst)
        return {
            "original": num_orig,
            "generated": 0,
            "removed": num_orig - target_count,
            "final": target_count
        }
    
    
    # Copiar todas las originales
    for src in src_images:
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copy2(src, dst)
    
    # Generar aumentadas hasta alcanzar target_count
    needed = target_count - num_orig
    print(f"      Generando {needed} aumentadas en {os.path.basename(dst_dir)}...")
    
    # Transformaciones con OpenCV
    def rotate(img, angle):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))
    
    def flip(img, code):
        return cv2.flip(img, code)  # 1 horizontal, 0 vertical, -1 ambos
    
    # Lista de transformaciones (nombre, función)
    transforms = [
        ('rot90', lambda x: rotate(x, 90)),
        ('rot180', lambda x: rotate(x, 180)),
        ('rot270', lambda x: rotate(x, 270)),
        ('fliplr', lambda x: flip(x, 1)),
        ('flipud', lambda x: flip(x, 0)),
        ('bright_plus30', lambda x: adjust_brightness(x, 30)),
        ('bright_minus30', lambda x: adjust_brightness(x, -30)),
    ]
    
    generated = 0
    idx = 0
    original_names = [os.path.splitext(os.path.basename(src))[0] for src in src_images]
    
    while generated < needed:
        for orig_name in original_names:
            if generated >= needed:
                break
            # Tomar una transformación cíclicamente
            tech_name, tech_func = transforms[idx % len(transforms)]
            # Buscar la imagen original correspondiente (la primera que coincida)
            src_path = [s for s in src_images if os.path.splitext(os.path.basename(s))[0] == orig_name][0]
            img = cv2.imread(src_path)
            if img is None:
                continue
            aug_img = tech_func(img)
            out_name = f"{orig_name}_{tech_name}.jpg"
            out_path = os.path.join(dst_dir, out_name)
            cv2.imwrite(out_path, aug_img)
            generated += 1
            idx += 1
            if generated % 20 == 0:
                print(f"         Progreso: {generated}/{needed}")
    print(f"      ✅ {os.path.basename(dst_dir)}: {num_orig} → {target_count}")
    return {
        "original": num_orig,
        "generated": needed,
        "removed": 0,
        "final": target_count
    }

# ===================================================
# PROCESAMIENTO PRINCIPAL
# ===================================================
def main():
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    report = []
    
    for class_name in os.listdir(INPUT_BASE):
        class_path = os.path.join(INPUT_BASE, class_name)
        if not os.path.isdir(class_path):
            continue
        
        print(f"\n📁 Procesando clase: {class_name}")
        structure_type, data = get_subclass_structure(class_path)
        
        if structure_type == 'flat':
            # Clase plana: todas las imágenes están en class_path
            images = data
            total_orig = len(images)
            print(f"   Total originales: {total_orig} | Objetivo: {TARGET_PER_CLASS}")
            dst_class = os.path.join(OUTPUT_BASE, class_name)
            stats = copy_and_augment(images, dst_class, TARGET_PER_CLASS)

            report.append({
                "Clase": class_name,
                "Subclase": "-",
                "Originales": stats["original"],
                "Generadas": stats["generated"],
                "Eliminadas": stats["removed"],
                "Final": stats["final"]
            })
        
        else:  # nested
            # Calcular total originales por subclase
            sub_counts = {sub: len(img_list) for sub, img_list in data.items()}
            total_orig = sum(sub_counts.values())
            print(f"   Total originales: {total_orig} | Objetivo: {TARGET_PER_CLASS}")
            
            # Distribuir el objetivo entre subclases proporcionalmente
            target_sub = {}
            for sub, cnt in sub_counts.items():
                target_sub[sub] = int(round(TARGET_PER_CLASS * cnt / total_orig))
            # Ajuste por redondeo para que la suma sea exactamente TARGET_PER_CLASS
            diff = TARGET_PER_CLASS - sum(target_sub.values())
            if diff != 0:
                # Sumar o restar a la subclase más grande
                largest = max(target_sub, key=target_sub.get)
                target_sub[largest] += diff
            
            # Procesar cada subclase
            for sub, target_cnt in target_sub.items():
                src_images = data[sub]
                dst_sub = os.path.join(OUTPUT_BASE, class_name, sub)
                print(f"   → Subclase '{sub}': {len(src_images)} → {target_cnt}")
                stats = copy_and_augment(src_images, dst_sub, target_cnt)

                report.append({
                    "Clase": class_name,
                    "Subclase": sub,
                    "Originales": stats["original"],
                    "Generadas": stats["generated"],
                    "Eliminadas": stats["removed"],
                    "Final": stats["final"]
                })
    
    print("\n🎉 Balanceo completado. Revisa la carpeta:", OUTPUT_BASE)
    csv_path = os.path.join(OUTPUT_BASE, "dataset_report.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Clase",
                "Subclase",
                "Originales",
                "Generadas",
                "Eliminadas",
                "Final"
            ]
        )

        writer.writeheader()

        writer.writerows(report)

    print(f"\n📄 Reporte guardado en: {csv_path}")

if __name__ == "__main__":
    main()