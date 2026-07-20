"""
DETECTOR DE IMÁGENES DUPLICADAS (mismo contenido, distintas dimensiones)
Uso: python detectar_duplicados.py
"""

import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from collections import defaultdict

# ------------------------------------------------------------
# 1. Recorte de bordes blancos (para que el fondo no interfiera)
# ------------------------------------------------------------
def crop_white_borders(image_path, white_threshold=250, padding=5):
    """Elimina el fondo blanco alrededor del objeto principal."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    x, y, w, h = cv2.boundingRect(coords)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2*padding)
    h = min(img.shape[0] - y, h + 2*padding)
    cropped = img[y:y+h, x:x+w]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

# ------------------------------------------------------------
# 2. Hash perceptual robusto (normaliza tamaño y elimina fondo blanco)
# ------------------------------------------------------------
def get_robust_hash(image_path, hash_size=16, crop_white=True, target_size=256):
    """
    Calcula una huella digital (pHash) de la imagen, después de:
    - Recortar fondo blanco (opcional)
    - Redimensionar a un cuadrado manteniendo aspecto (sin agrandar imágenes pequeñas)
    """
    if crop_white:
        pil_img = crop_white_borders(image_path)
        if pil_img is None:
            pil_img = Image.open(image_path)
    else:
        pil_img = Image.open(image_path)
    
    pil_img = pil_img.convert('RGB')
    w, h = pil_img.size
    
    # Solo reducir si es más grande que target_size (no agrandar)
    if w > target_size or h > target_size:
        pil_img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Centrar sobre lienzo negro de tamaño target_size x target_size
    new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    w2, h2 = pil_img.size
    x_offset = (target_size - w2) // 2
    y_offset = (target_size - h2) // 2
    new_img.paste(pil_img, (x_offset, y_offset))
    
    return imagehash.phash(new_img, hash_size=hash_size)

# ------------------------------------------------------------
# 3. Agrupar imágenes duplicadas en una carpeta
# ------------------------------------------------------------
def find_duplicates(folder_path, hash_size=16, threshold=10, crop_white=True):
    """
    Escanea una carpeta y devuelve listas de rutas de imágenes duplicadas.
    threshold: distancia de Hamming máxima para considerar dos imágenes iguales.
    """
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    hash_to_paths = defaultdict(list)
    
    # Calcular hash de cada imagen
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            filepath = os.path.join(folder_path, filename)
            try:
                img_hash = get_robust_hash(filepath, hash_size, crop_white)
                hash_to_paths[img_hash].append(filepath)
                print(f"Procesada: {filename} -> Hash: {img_hash}")
            except Exception as e:
                print(f"Error con {filename}: {e}")
    
    # Agrupar hashes cercanos (no solo exactamente iguales)
    unique_hashes = list(hash_to_paths.keys())
    merged_groups = []
    used = set()
    
    for i, h1 in enumerate(unique_hashes):
        if i in used:
            continue
        group = hash_to_paths[h1][:]
        used.add(i)
        for j in range(i+1, len(unique_hashes)):
            if j in used:
                continue
            h2 = unique_hashes[j]
            if h1 - h2 <= threshold:
                group.extend(hash_to_paths[h2])
                used.add(j)
        if len(group) > 1:
            merged_groups.append(group)
    
    return merged_groups

# ------------------------------------------------------------
# 4. Mostrar resultados en consola
# ------------------------------------------------------------
def mostrar_resultados(grupos_duplicados):
    if not grupos_duplicados:
        print("✅ No se encontraron imágenes duplicadas.")
        return
    
    print(f"\n🔍 Se encontraron {len(grupos_duplicados)} grupo(s) de imágenes duplicadas:\n")
    for idx, grupo in enumerate(grupos_duplicados, 1):
        print(f"📁 Grupo {idx}:")
        for ruta in grupo:
            print(f"   - {os.path.basename(ruta)}")
        print()

# ------------------------------------------------------------
# 5. CONFIGURACIÓN Y EJECUCIÓN PRINCIPAL
# ------------------------------------------------------------
if __name__ == "__main__":
    # ===== CAMBIA ESTA RUTA =====
    #CARPETA_IMAGENES = "/home/dao/Documentos/proyectosML/rfv2/images/plastic"   # <--- Pon aquí la ruta de tu carpeta
    #"/home/dao/Documentos/proyectosML/rfv2/cardboard_prueba"
    CARPETA_IMAGENES = "/home/dao/Documentos/proyectosML/ml_tf/images_2026/images/metal"
    
    # Parámetros ajustables
    UMBRAL_DISTANCIA = 20    # 5 = muy estricto, 15 = tolerante
    RECORTAR_BLANCO = True     # Recomendado True para eliminar fondos blancos
    
    print(f"Escaneando carpeta: {CARPETA_IMAGENES}")
    duplicados = find_duplicates(CARPETA_IMAGENES, 
                                 hash_size=16, 
                                 threshold=UMBRAL_DISTANCIA, 
                                 crop_white=RECORTAR_BLANCO)
    mostrar_resultados(duplicados)


    # librerias : pip install opencv-python numpy Pillow ImageHash#