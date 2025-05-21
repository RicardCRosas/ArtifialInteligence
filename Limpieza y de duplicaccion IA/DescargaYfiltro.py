# Proyecto: Descarga, limpieza y deduplicación de imágenes con Big Data
# Autor: Ricardo Carballido Rosas
import openai
import os
import time
import shutil
import json
import hashlib
import jsonschema
import requests
import imagehash
import urllib.request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms

from PIL import Image, ImageStat
from io import BytesIO
from tqdm import tqdm
from collections import Counter, defaultdict

from pyspark.sql import SparkSession
from datasketch import MinHash, MinHashLSH
import torch
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException
from serpapi import GoogleSearch
from collections import defaultdict


# =================== PARÁMETROS ===================
TRADUCCIONES = {

    "grietas": [
        "cracks",
        "grietas",
        "Risse",
        "crepe"
    ]
}

CANTIDAD_OBJETIVO = 1000
TAM_MINIMO = (640, 640)
PHASH_SIZE = 8
NUM_PERM = 128
LSH_THRESHOLD = 0.90
JACCARD_MINIMO = 0.01
PAUSA_SEGUNDOS = 10
MAX_RESULTADOS_POR_RONDA = 1000
MAX_RONDAS = 10

INPUT_DIR = "dataset_unificado"
OUTPUT_DIR = "dataset_limpio"
DUPLICATES_DIR = "duplicados_por_clase"
CLASSES_FILE = "imagenet_classes.txt"
SCHEMA_FILE = "movies.schema.json"
JSON_OUTPUT = "resumen.json"
JSON_DUPLICADOS = "imagenes_duplicadas.json"
JSON_ANALISIS = "analisis_dataset.json"
MODO_SIMULACION = False

etiquetas_no_deseadas_keywords = [
    'web', 'site', 'jacket', 'comic', 'book', 'envelope', 'curtain',
    'coil', 'slide', 'powder', 'menu', 'shade'
]

# =================== INICIALIZACIÓN ===================
spark = SparkSession.builder.appName("LimpiezaDatasetIA").getOrCreate()
print("✅ SparkSession iniciada")

modelo_clasificador = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
modelo_clasificador.eval()
transformador = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if not os.path.exists(CLASSES_FILE):
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    urllib.request.urlretrieve(url, CLASSES_FILE)

with open(CLASSES_FILE, "r") as f:
    etiquetas_imagenet = [line.strip() for line in f.readlines()]

# =================== FUNCIONES ===================
hashes_existentes = set()

def hash_imagen(imagen):
    return hashlib.md5(imagen.tobytes()).hexdigest()

def descargar_dataset_multilingue(etiquetas, carpeta_destino):
    os.makedirs(carpeta_destino, exist_ok=True)
    descargadas = 0
    rondas = 0

    for etiqueta in etiquetas:
        while descargadas < CANTIDAD_OBJETIVO and rondas < MAX_RONDAS:
            rondas += 1
            try:
                with DDGS() as ddgs:
                    resultados = ddgs.images(keywords=etiqueta, max_results=MAX_RESULTADOS_POR_RONDA)
                    for resultado in tqdm(resultados, desc=f"[{etiqueta}] Ronda {rondas}"):
                        if descargadas >= CANTIDAD_OBJETIVO:
                            break
                        try:
                            url = resultado["image"]
                            response = requests.get(url, timeout=5)
                            image = Image.open(BytesIO(response.content)).convert("RGB")
                            if image.size[0] < TAM_MINIMO[0] or image.size[1] < TAM_MINIMO[1]:
                                continue
                            h = hash_imagen(image)
                            if h in hashes_existentes:
                                continue
                            hashes_existentes.add(h)
                            nombre = f"{etiqueta.replace(' ', '_').lower()}_{descargadas+1}.jpg"
                            ruta = os.path.join(carpeta_destino, nombre)
                            image.save(ruta, "JPEG")
                            descargadas += 1
                        except:
                            continue
            except RatelimitException:
                print(f"⚠️ DuckDuckGo bloqueó temporalmente la IP. Esperando {PAUSA_SEGUNDOS} segundos...")
                time.sleep(PAUSA_SEGUNDOS)



def descargar_dataset_serpapi(etiquetas, carpeta_destino, api_key):
    os.makedirs(carpeta_destino, exist_ok=True)
    descargadas = 0

    for etiqueta in etiquetas:
        params = {
            "engine": "google",
            "q": etiqueta,
            "tbm": "isch",
            "ijn": "0",
            "api_key": api_key
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            imagenes = results.get("images_results", [])

            for resultado in tqdm(imagenes, desc=f"[{etiqueta}]"):
                if descargadas >= CANTIDAD_OBJETIVO:
                    break
                try:
                    url = resultado["original"]
                    response = requests.get(url, timeout=5)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    if image.size[0] < TAM_MINIMO[0] or image.size[1] < TAM_MINIMO[1]:
                        continue
                    h = hash_imagen(image)
                    if h in hashes_existentes:
                        continue
                    hashes_existentes.add(h)
                    nombre = f"{etiqueta.replace(' ', '_').lower()}_{descargadas+1}.jpg"
                    ruta = os.path.join(carpeta_destino, nombre)
                    image.save(ruta, "JPEG")
                    descargadas += 1
                except:
                    continue
        except Exception as e:
            print(f"❌ Error SerpAPI con '{etiqueta}': {e}")

def descargar_dataset_google(etiquetas, carpeta_destino, api_key, cse_id):
    os.makedirs(carpeta_destino, exist_ok=True)
    descargadas = 0

    for etiqueta in etiquetas:
        print(f"🔍 [Google] Buscando: {etiqueta}")
        for start in range(1, 100, 10):  # hasta 100 resultados, 10 por página
            if descargadas >= CANTIDAD_OBJETIVO:
                break
            params = {
                "q": etiqueta,
                "cx": cse_id,
                "key": api_key,
                "searchType": "image",
                "num": 10,
                "start": start
            }
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
            if response.status_code != 200:
                print(f"❌ Error Google API: {response.text}")
                break

            results = response.json().get("items", [])
            for item in tqdm(results, desc=f"[Google] {etiqueta}"):
                try:
                    url = item["link"]
                    image_data = requests.get(url, timeout=5).content
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    if image.size[0] < TAM_MINIMO[0] or image.size[1] < TAM_MINIMO[1]:
                        continue
                    h = hash_imagen(image)
                    if h in hashes_existentes:
                        continue
                    hashes_existentes.add(h)
                    nombre = f"{etiqueta.replace(' ', '_').lower()}_{descargadas+1}.jpg"
                    ruta = os.path.join(carpeta_destino, nombre)
                    image.save(ruta, "JPEG")
                    descargadas += 1
                    if descargadas >= CANTIDAD_OBJETIVO:
                        break
                except:
                    continue



def generar_imagenes_dalle(clase, carpeta_destino, api_key, cantidad=5):
    os.makedirs(carpeta_destino, exist_ok=True)
    openai.api_key = api_key

    for i in range(1, cantidad + 1):
        prompt = f"high-resolution photo of {clase}, industrial context, clean background"
        try:
            response = openai.Image.create(prompt=prompt, n=1, size="512x512")
            url = response["data"][0]["url"]
            image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
            nombre = f"{clase.replace(' ', '_')}_dalle_{i}.png"
            ruta = os.path.join(carpeta_destino, nombre)
            image.save(ruta, "PNG")
        except Exception as e:
            print(f"❌ Error generando con DALL·E ({clase}): {e}")


def copiar_imagenes_existentes(etiqueta, carpeta_destino):
    nombre_dataset = "dataset_" + etiqueta.replace(" ", "_").lower()
    carpeta_origen = os.path.join("C:/Users/Ricardo/PycharmProjects/ArtifialInteligence/.ipynb_checkpoints", nombre_dataset)

    if not os.path.exists(carpeta_origen):
        print(f"⚠️ Carpeta no encontrada: {carpeta_origen}")
        return

    for archivo in os.listdir(carpeta_origen):
        if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
            origen = os.path.join(carpeta_origen, archivo)
            destino = os.path.join(carpeta_destino, archivo)
            try:
                shutil.copy(origen, destino)
            except Exception as e:
                print(f"❌ Error copiando {archivo}: {e}")



def obtener_etiquetas_principales(path, top=3):
    etiquetas_no_deseadas_keywords = ['web', 'site', 'jacket', 'comic', 'book', 'envelope', 'curtain', 'coil', 'slide', 'powder', 'menu', 'shade']
    try:
        imagen = Image.open(path).convert("RGB")
        tensor = transformador(imagen).unsqueeze(0)
        with torch.no_grad():
            salida = modelo_clasificador(tensor)
            probs = torch.nn.functional.softmax(salida[0], dim=0)
            top_idxs = torch.topk(probs, top).indices
            return [etiquetas_imagenet[idx] for idx in top_idxs if not any(p in etiquetas_imagenet[idx].lower() for p in etiquetas_no_deseadas_keywords)]
    except Exception as e:
        print(f"❌ Error clasificando {path}: {e}")
        return []

def procesar_imagen(path):
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path).convert("RGB")
        phash = imagehash.phash(img, hash_size=PHASH_SIZE)
        return (path, str(phash))
    except Exception as e:
        print(f"❌ Error con {path}: {e}")
        return None

def hash_a_minhash(phash_str):
    mh = MinHash(num_perm=NUM_PERM)
    for i in phash_str:
        mh.update(i.encode('utf-8'))
    return mh

def imagen_valida_por_calidad(path):
    try:
        img = Image.open(path).convert("L")
        stat = ImageStat.Stat(img)
        return stat.stddev[0] > 5
    except:
        return False

def filtrar_por_embedding_resnet(paths_limpias, threshold=0.70):
    print("🚀 Iniciando filtrado por embedding con ResNet...")

    # Cargar modelo
    modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    modelo.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)

    transformador = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    embeddings = []
    rutas_validas = []

    for path in tqdm(paths_limpias, desc="🧠 Generando embeddings"):
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = transformador(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = modelo(img_tensor).cpu().numpy().flatten()
            embeddings.append(emb)
            rutas_validas.append(path)
        except Exception as e:
            print(f"❌ Error en {path}: {e}")

    # Calcular centroide
    embeddings = np.array(embeddings)
    centroide = np.mean(embeddings, axis=0).reshape(1, -1)
    similitudes = cosine_similarity(embeddings, centroide).flatten()

    # Filtrar por similitud
    rutas_filtradas = [rutas_validas[i] for i in range(len(rutas_validas)) if similitudes[i] >= threshold]
    print(f"✅ Imágenes después del filtrado por embedding: {len(rutas_filtradas)} de {len(rutas_validas)}")

    return rutas_filtradas


# =================== DESCARGA ===================
# # para contar todo se hace con ctrl+/
# for clase, etiquetas in TRADUCCIONES.items():
#     carpeta_destino = os.path.join(INPUT_DIR, clase.replace(" ", "_").lower())
#     os.makedirs(carpeta_destino, exist_ok=True)  # Garantiza que la carpeta existe

    # # Paso 1: Descargar imágenes nuevas con SerpAPI
    # # crea tu cuenta aqui https://serpapi.com/users/sign_up y consigue tu clave
    # print(f"🔽 Descargando con SerpAPI: {clase}")
    # descargar_dataset_serpapi(etiquetas, carpeta_destino=carpeta_destino, api_key="1d2568014d235c0737b447b1f72c8b7664775063f2a6d7c43a41ad66bf25e891")
    # print(f"🕒 Esperando {PAUSA_SEGUNDOS} segundos...\n")
    # time.sleep(PAUSA_SEGUNDOS)
    # #
    # # # Paso 2: Descargar imágenes nuevas con DuckDuckGo
    # print(f"🔽 Descargando con DuckDuckGo: {clase}")
    # descargar_dataset_multilingue(etiquetas, carpeta_destino=carpeta_destino)
    # print(f"🕒 Esperando {PAUSA_SEGUNDOS} segundos...\n")
    # time.sleep(PAUSA_SEGUNDOS)
    # #
    # #
    # # # Paso 3: Descargar imágenes con Google Custom Search
    # print(f"🔽 Descargando con Google Search API: {clase}")
    # descargar_dataset_google(etiquetas, carpeta_destino=carpeta_destino,
    #                          api_key="AIzaSyB9ERRVQEEIZEDoUrA8QlJQCKS5VJaTE6g", cse_id="f44a3c7beee6b4466")
    # print(f"🕒 Esperando {PAUSA_SEGUNDOS} segundos...\n")
    # time.sleep(PAUSA_SEGUNDOS)
    # #
    # # # Paso 4: Generar imágenes con DALL·E
    # print(f"🎨 Generando con DALL·E: {clase}")
    # generar_imagenes_dalle(clase, carpeta_destino=carpeta_destino,
    #                        api_key="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", cantidad=5)
    # print(f"🕒 Esperando {PAUSA_SEGUNDOS} segundos...\n")
    # time.sleep(PAUSA_SEGUNDOS)
    #
    # # Paso 5: Copiar imágenes ya existentes desde dataset_{nombre}
    # nombre_dataset_local = "dataset_" + clase.replace(" ", "_").lower()
    # ruta_dataset_local = os.path.join("C:/Users/Ricardo/PycharmProjects/ArtifialInteligence/.ipynb_checkpoints", nombre_dataset_local)
    #
    # if os.path.exists(ruta_dataset_local):
    #     for archivo in os.listdir(ruta_dataset_local):
    #         if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
    #             origen = os.path.join(ruta_dataset_local, archivo)
    #             destino = os.path.join(carpeta_destino, archivo)
    #             try:
    #                 if not os.path.exists(destino):  # No sobreescribir si ya existe
    #                     shutil.copy(origen, destino)
    #             except Exception as e:
    #                 print(f"❌ Error copiando {archivo}: {e}")
    # else:
    #     print(f"⚠️ No se encontró: {ruta_dataset_local}")





# =================== LIMPIEZA ===================
# =================== LIMPIEZA OPTIMIZADA ===================

# 1. Eliminar y recrear directorios de salida
for dir_path in [OUTPUT_DIR, DUPLICATES_DIR]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

# 2. Recopilar imágenes válidas desde INPUT_DIR
paths = []
for root, _, files in os.walk(INPUT_DIR):
    for name in files:
        if name.lower().endswith(('.png', '.jpg', '.jpeg')):
            paths.append(os.path.join(root, name))

# 3. Procesar imágenes: obtener pHash y validar apertura
procesadas = []
for p in paths:
    try:
        img = Image.open(p).convert("RGB")
        h = hash_imagen(img)
        procesadas.append((p, h))
    except Exception as e:
        print(f"Error al procesar {p}: {e}")

# 4. Crear índice LSH con MinHash
lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
hash_dict = {}
for path, phash_str in procesadas:
    mh = hash_a_minhash(phash_str)
    if phash_str not in hash_dict:
        hash_dict[phash_str] = mh
        lsh.insert(phash_str, mh)

# 5. Identificar duplicados
hash_grupos = defaultdict(list)
for path, phash_str in procesadas:
    hash_grupos[phash_str].append(path)
duplicados_dict = {k: v for k, v in hash_grupos.items() if len(v) > 1}
imagenes_duplicadas_set = {p for grupo in duplicados_dict.values() for p in grupo[1:]}

# Guardar duplicados detectados
with open(JSON_DUPLICADOS, "w") as f:
    json.dump(duplicados_dict, f, indent=4)

# Copiar duplicados a carpeta de análisis
for grupo in duplicados_dict.values():
    for path in grupo[1:]:
        clase = os.path.basename(os.path.dirname(path)) or "desconocida"
        destino = os.path.join(DUPLICATES_DIR, clase, os.path.basename(path))
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        shutil.copy(path, destino)

# 6. Análisis de imágenes limpias
phashes_limpias = [(p, h) for p, h in procesadas if p not in imagenes_duplicadas_set]
referencia_minhashes = [(p, hash_a_minhash(h)) for p, h in phashes_limpias]
# Solo los paths de imágenes limpias (sin duplicados, buena calidad, etc.)
paths_limpias_sin_duplicados = [p for p, _ in phashes_limpias]

# 🔍 Aplicar filtrado por embedding (opcional, umbral ajustable)
paths_finales = filtrar_por_embedding_resnet(paths_limpias_sin_duplicados, threshold=0.20)

# Actualizar phashes_limpias para usar solo las que pasaron el filtro
phashes_limpias = [(p, h) for p, h in phashes_limpias if p in paths_finales]

# 7. Validar y copiar imágenes limpias
resumen_valido = []
with open(SCHEMA_FILE, "r") as schemafile:
    schema = json.load(schemafile)

for path, phash_str in phashes_limpias:
    mh_test = hash_a_minhash(phash_str)
    similitudes = [mh_test.jaccard(ref_mh) for _, ref_mh in referencia_minhashes if path != _]

    if not similitudes or max(similitudes) < JACCARD_MINIMO:
        continue
    if not imagen_valida_por_calidad(path):
        continue

    clases_visuales = obtener_etiquetas_principales(path)
    if not clases_visuales:
        continue

    clase = os.path.basename(os.path.dirname(path))
    nombre = os.path.basename(path)
    nueva_ruta = os.path.join(OUTPUT_DIR, clase)
    os.makedirs(nueva_ruta, exist_ok=True)
    shutil.copy(path, os.path.join(nueva_ruta, nombre))

    entrada = {
        "movie": {
            "title": nombre,
            "year": 2025,
            "director": clase,
            "clases_visuales": clases_visuales
        }
    }

    try:
        jsonschema.validate(entrada, schema)
        resumen_valido.append(entrada)
    except jsonschema.ValidationError as e:
        print(f"❌ JSON inválido para {nombre}: {e.message}")

# 8. Guardar resumen de imágenes válidas
with open(JSON_OUTPUT, "w") as f:
    json.dump(resumen_valido, f, indent=4)

print(f"✅ Proceso de limpieza completado: {len(resumen_valido)} imágenes limpias guardadas.")

# =================== ANÁLISIS POR CLASE ===================

analisis = {}
for path, _ in procesadas:
    clase = os.path.basename(os.path.dirname(path)) or "desconocida"
    analisis.setdefault(clase, {"total": 0, "duplicadas": 0, "limpias": 0})
    if path in imagenes_duplicadas_set:
        analisis[clase]["duplicadas"] += 1
    else:
        analisis[clase]["limpias"] += 1
    analisis[clase]["total"] += 1

with open(JSON_ANALISIS, "w") as f:
    json.dump(analisis, f, indent=4)

print("📊 Análisis por clase guardado en 'analisis_dataset.json'.")
