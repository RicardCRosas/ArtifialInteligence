import os
import time
import requests
from PIL import Image
from io import BytesIO
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException
from tqdm import tqdm

# Lista de clases
CLASES = [
    "rail electrified monorail system",
    "copper bus bar",
    "corroded copper",
    "current collector sliding contact collector",
    "cobre quemado",
    "grietas",
]

# Parámetros
CANTIDAD_OBJETIVO = 1000
TAM_MINIMO = (640, 640)
PAUSA_SEGUNDOS = 10
MAX_RESULTADOS_POR_RONDA = 1000
MAX_RONDAS = 10  # evitar bucle infinito

def descargar_dataset(objeto, carpeta_destino="dataset", cantidad=1000, min_size=(640, 640)):
    os.makedirs(carpeta_destino, exist_ok=True)
    nombres_existentes = set(os.listdir(carpeta_destino))
    descargadas = len(nombres_existentes)

    if descargadas >= cantidad:
        print(f"⏩ Ya existen {descargadas} imágenes para '{objeto}', saltando.")
        return descargadas

    descartadas_por_tamano = 0
    fallidas = 0
    intento_ratelimit = 0
    rondas = 0

    while descargadas < cantidad and rondas < MAX_RONDAS:
        rondas += 1
        try:
            with DDGS() as ddgs:
                resultados = ddgs.images(keywords=objeto, max_results=MAX_RESULTADOS_POR_RONDA)

                for resultado in tqdm(resultados, total=MAX_RESULTADOS_POR_RONDA, desc=f"[{objeto}] Ronda {rondas}"):
                    if descargadas >= cantidad:
                        break
                    try:
                        url = resultado["image"]
                        response = requests.get(url, timeout=5)
                        image = Image.open(BytesIO(response.content))

                        if image.mode not in ("RGB", "RGBA"):
                            continue
                        if image.size[0] < min_size[0] or image.size[1] < min_size[1]:
                            descartadas_por_tamano += 1
                            continue

                        nombre = f"{objeto.replace(' ', '_').lower()}_{descargadas+1}.jpg"
                        ruta = os.path.join(carpeta_destino, nombre)
                        if nombre in nombres_existentes:
                            continue  # evitar duplicados por nombre
                        image.convert("RGB").save(ruta, "JPEG")
                        nombres_existentes.add(nombre)
                        descargadas += 1
                    except Exception:
                        fallidas += 1
                        continue

        except RatelimitException:
            intento_ratelimit += 1
            print(f"⚠️ DuckDuckGo bloqueó temporalmente la IP (intento #{intento_ratelimit}). Esperando {PAUSA_SEGUNDOS} segundos...")
            time.sleep(PAUSA_SEGUNDOS)

    print(f"\n✅ Descargadas {descargadas}/{cantidad} imágenes válidas para '{objeto}' en '{carpeta_destino}'")
    print(f"📉 Imágenes descartadas por tamaño: {descartadas_por_tamano}")
    print(f"❌ Imágenes fallidas: {fallidas}")
    if descargadas < cantidad:
        print(f"⚠️ No se alcanzaron las {cantidad} imágenes válidas para '{objeto}' tras {rondas} rondas.")
    return descargadas

# Ejecutar para todas las clases
for clase in CLASES:
    carpeta = f"dataset_{clase.replace(' ', '_').lower()}"
    descargar_dataset(clase, carpeta_destino=carpeta, cantidad=CANTIDAD_OBJETIVO)
    print(f"🕒 Esperando {PAUSA_SEGUNDOS} segundos antes de la siguiente clase...\n")
    time.sleep(PAUSA_SEGUNDOS)

print("\n✅ Todas las clases fueron procesadas correctamente.")
