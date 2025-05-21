import os

# Rutas principales de las carpetas
carpetas_principales = [
    "C:/Users/Ricardo/PycharmProjects/ArtifialInteligence/Limpieza y de duplicaccion IA/dataset_unificado",
    "C:/Users/Ricardo/PycharmProjects/ArtifialInteligence/Limpieza y de duplicaccion IA/dataset_limpio",
    "C:/Users/Ricardo/PycharmProjects/ArtifialInteligence/Limpieza y de duplicaccion IA/dataset_yolo_segment"
]

# Extensiones válidas de imágenes
extensiones_validas = ('.jpg', '.jpeg', '.png', '.webp')

def contar_imagenes_por_subcarpeta(carpeta_raiz):
    conteo_total = 0
    print(f"\n📂 Análisis de subcarpetas en: {carpeta_raiz}")
    for subcarpeta, _, archivos in os.walk(carpeta_raiz):
        conteo = sum(1 for archivo in archivos if archivo.lower().endswith(extensiones_validas))
        if conteo > 0:
            print(f"   📁 {subcarpeta} → {conteo} imagen(es)")
        conteo_total += conteo
    print(f"✅ Total en {carpeta_raiz}: {conteo_total} imagen(es)")
    return conteo_total

# Recuento global
total_global = 0
for carpeta in carpetas_principales:
    total_global += contar_imagenes_por_subcarpeta(carpeta)

print(f"\n📊 TOTAL GLOBAL DE IMÁGENES EN TODAS LAS CARPETAS: {total_global}")
