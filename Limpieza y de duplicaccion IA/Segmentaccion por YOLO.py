import os
import cv2
from ultralytics import YOLO

# === CONFIGURACIÓN ===
modelo_path = r"C:/Users/Ricardo/PycharmProjects/ArtifialInteligence/Limpieza y de duplicaccion IA/runs/segment/train9/weights/best.pt"
carpeta_origen = r"C:\Users\Ricardo\Downloads\dataset_limpio"
carpeta_salida = r"C:/Users/Ricardo/PycharmProjects/ArtifialInteligence/Limpieza y de duplicaccion IA/dataset_yolo_segment"

# === DICCIONARIO DE CLASES ===
id2nombre = {
    0: "burned_copper",
    1: "copper_bus_bar",
    2: "corroded_copper",
    3: "crack",
    4: "current_collector_sliding_contact_collector",
    5: "rail-electrified-monorail-system"
}

# === CREAR CARPETAS DE SALIDA ===
imagenes_salida = os.path.join(carpeta_salida, "images")
labels_salida = os.path.join(carpeta_salida, "labels")
os.makedirs(imagenes_salida, exist_ok=True)
os.makedirs(labels_salida, exist_ok=True)

# === CARGAR MODELO ===
modelo = YOLO(modelo_path)

# === PROCESAR TODAS LAS IMÁGENES ===
for root, _, files in os.walk(carpeta_origen):
    for archivo in files:
        if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
            ruta_img = os.path.join(root, archivo)
            try:
                img = cv2.imread(ruta_img)
                if img is None:
                    print(f"⚠️ No se pudo leer: {ruta_img}")
                    continue

                img_resized = cv2.resize(img, (640, 640))
                nombre_base = os.path.splitext(archivo)[0]
                ruta_img_salida = os.path.join(imagenes_salida, f"{nombre_base}.jpg")
                ruta_txt_salida = os.path.join(labels_salida, f"{nombre_base}.txt")

                # Guardar imagen redimensionada
                cv2.imwrite(ruta_img_salida, img_resized)

                # Ejecutar segmentación
                resultados = modelo(img_resized, verbose=False)[0]

                if resultados.masks is None or resultados.boxes is None:
                    print(f"⚠️ Sin detecciones en {archivo}")
                    continue

                clases = resultados.boxes.cls.tolist()
                segmentos = resultados.masks.xyn  # coordenadas normalizadas

                if not clases or not segmentos:
                    print(f"⚠️ Sin clases o segmentos en {archivo}")
                    continue

                with open(ruta_txt_salida, "w") as f:
                    for clase_id, segmento in zip(clases, segmentos):
                        clase_id_int = int(clase_id)
                        if clase_id_int not in id2nombre:
                            print(f"⚠️ Clase desconocida: {clase_id_int}")
                            continue
                        puntos = " ".join([f"{x:.6f} {y:.6f}" for x, y in segmento])
                        f.write(f"{clase_id_int} {puntos}\n")

                print(f"✅ Segmentado y anotado: {archivo}")

            except Exception as e:
                print(f"❌ Error con {archivo}: {e}")
