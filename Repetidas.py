# Proyecto: Limpieza y deduplicación de imágenes con Big Data
# Autor: Ricardo Carballido Rosas

import os
import shutil
import imagehash
import json
import jsonschema
from PIL import Image, ImageStat
from pyspark.sql import SparkSession
from datasketch import MinHash, MinHashLSH
from torchvision import models, transforms
import torch
from collections import Counter, defaultdict
import urllib.request

# === Inicialización de Spark ===
spark = SparkSession.builder.appName("LimpiezaDatasetIA").getOrCreate()
sc = spark.sparkContext
print("✅ SparkSession iniciada")

# === Parámetros ===
INPUT_DIR = "C:/Users/Ricardo/PycharmProjects/ArtifialInteligence/.ipynb_checkpoints"
OUTPUT_DIR = "C:/Users/Ricardo/PycharmProjects/ArtifialInteligence/dataset_limpio"
DUPLICATES_DIR = "C:/Users/Ricardo/PycharmProjects/ArtifialInteligence/duplicados_por_clase"
PHASH_SIZE = 8
NUM_PERM = 128
LSH_THRESHOLD = 0.90
JACCARD_MINIMO = 0.01
SCHEMA_FILE = "/Limpieza y de duplicaccion IA/movies.schema.json"
JSON_OUTPUT = "resumen.json"
JSON_DUPLICADOS = "imagenes_duplicadas.json"
JSON_ANALISIS = "analisis_dataset.json"
CLASSES_FILE = "imagenet_classes.txt"
MODO_SIMULACION = False

from torchvision.models import MobileNet_V2_Weights
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

def obtener_etiquetas_principales(path, top=3):
    etiquetas_no_deseadas_keywords = ['web', 'site', 'jacket', 'comic', 'book', 'envelope', 'curtain', 'coil', 'slide', 'powder', 'menu', 'shade']
    try:
        imagen = Image.open(path).convert("RGB")
        tensor = transformador(imagen).unsqueeze(0)
        with torch.no_grad():
            salida = modelo_clasificador(tensor)
            probs = torch.nn.functional.softmax(salida[0], dim=0)
            top_idxs = torch.topk(probs, top).indices
            etiquetas_filtradas = []
            for idx in top_idxs:
                etiqueta = etiquetas_imagenet[idx]
                if not any(palabra in etiqueta.lower() for palabra in etiquetas_no_deseadas_keywords):
                    etiquetas_filtradas.append(etiqueta)
            return etiquetas_filtradas
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
        stddev = stat.stddev[0]
        return stddev > 5
    except:
        return False

for dir_path in [OUTPUT_DIR, DUPLICATES_DIR]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

paths = []
for root, _, files in os.walk(INPUT_DIR):
    for name in files:
        if name.lower().endswith(('.png', '.jpg', '.jpeg')):
            paths.append(os.path.join(root, name))

print(f"🔍 Total de imágenes encontradas: {len(paths)}")
procesadas = [r for r in (procesar_imagen(p) for p in paths) if r is not None]
print(f"📸 Imágenes válidas procesadas: {len(procesadas)}")

lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
hash_dict = {}
path_por_hash = {}
for path, phash_str in procesadas:
    mh = hash_a_minhash(phash_str)
    if phash_str not in path_por_hash:
        path_por_hash[phash_str] = path
        hash_dict[phash_str] = mh
        lsh.insert(phash_str, mh)

hash_grupos = {}
for path, phash_str in procesadas:
    hash_grupos.setdefault(phash_str, []).append(path)

duplicados_dict = {k: v for k, v in hash_grupos.items() if len(v) > 1}

with open(JSON_DUPLICADOS, "w") as f:
    json.dump(duplicados_dict, f, indent=4)
print(f"📄 JSON generado con duplicados: {len(duplicados_dict)} grupos en '{JSON_DUPLICADOS}'")

imagenes_duplicadas_set = set()
for phash, paths in duplicados_dict.items():
    for path in paths[1:]:
        imagenes_duplicadas_set.add(path)
        clase = os.path.basename(os.path.dirname(path)) or "desconocida"
        nombre = os.path.basename(path)
        dup_path = os.path.join(DUPLICATES_DIR, clase)
        os.makedirs(dup_path, exist_ok=True)
        shutil.copy(path, os.path.join(dup_path, nombre))

analisis = {}
for path, _ in procesadas:
    clase = os.path.basename(os.path.dirname(path)) or "desconocida"
    analisis.setdefault(clase, {"total": 0, "duplicadas": 0, "limpias": 0})
    if path in imagenes_duplicadas_set:
        analisis[clase]["duplicadas"] += 1
    else:
        analisis[clase]["limpias"] += 1
    analisis[clase]["total"] += 1

print("\n📊 Análisis por clase:")
for clase, stats in analisis.items():
    print(f"📁 {clase}: Total={stats['total']} | Duplicadas={stats['duplicadas']} | Limpias={stats['limpias']}")

with open(JSON_ANALISIS, "w") as f:
    json.dump(analisis, f, indent=4)

resumen = []
phashes_limpias = [(p, h) for p, h in procesadas if p not in imagenes_duplicadas_set]
referencia_minhashes = [(p, hash_a_minhash(h)) for p, h in phashes_limpias]

clases_por_directorio = defaultdict(list)
for path, _ in phashes_limpias:
    etiquetas = obtener_etiquetas_principales(path)
    clase = os.path.basename(os.path.dirname(path)) or "desconocida"
    clases_por_directorio[clase].extend(etiquetas)

for clase, etiquetas in clases_por_directorio.items():
    top5 = [cl for cl, _ in Counter(etiquetas).most_common(3)]
    print(f"🔎 {clase} → Clases visuales dominantes: {top5}")

for path, phash_str in phashes_limpias:
    mh_test = hash_a_minhash(phash_str)
    similitudes = [mh_test.jaccard(ref_mh) for _, ref_mh in referencia_minhashes if path != _]
    if not similitudes or max(similitudes) < JACCARD_MINIMO:
        print(f"🗑️ Eliminada por no parecerse al dataset: {path}")
        continue
    if not imagen_valida_por_calidad(path):
        print(f"🗑️ Eliminada por baja calidad visual: {path}")
        continue
    clase = os.path.basename(os.path.dirname(path)) or "desconocida"
    nombre = os.path.basename(path)
    nueva_ruta = os.path.join(OUTPUT_DIR, clase)
    os.makedirs(nueva_ruta, exist_ok=True)
    if MODO_SIMULACION:
        print(f"🧪 Simulación: se copiaría {path}")
    else:
        shutil.copy(path, os.path.join(nueva_ruta, nombre))
    resumen.append({"movie": {"title": nombre, "year": 2025, "director": clase}})

with open(SCHEMA_FILE, "r") as schemafile:
    schema = json.load(schemafile)

resumen_valido = []
for entry in resumen:
    try:
        jsonschema.validate(entry, schema)
        resumen_valido.append(entry)
    except jsonschema.ValidationError as e:
        print(f"❌ JSON inválido para {entry['movie']['title']}: {e.message}")

with open(JSON_OUTPUT, "w") as f:
    json.dump(resumen_valido, f, indent=4)

print(f"✅ Proceso completo. {len(resumen_valido)} imágenes limpias guardadas en '{OUTPUT_DIR}'.")
