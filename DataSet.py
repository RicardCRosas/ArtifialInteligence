import os
import requests
from PIL import Image
from io import BytesIO
from duckduckgo_search import DDGS
from tqdm import tqdm

def descargar_dataset(objeto, carpeta_destino="dataset", cantidad=500, min_size=(128, 128)):
    os.makedirs(carpeta_destino, exist_ok=True)
    descargadas = 0

    with DDGS() as ddgs:
        resultados = ddgs.images(keywords=objeto, max_results=cantidad * 2)

        for i, resultado in enumerate(tqdm(resultados, total=cantidad * 2)):
            if descargadas >= cantidad:
                break
            try:
                url = resultado["image"]
                response = requests.get(url, timeout=5)
                image = Image.open(BytesIO(response.content))

                # Validar formato y tamaño
                if image.mode not in ("RGB", "RGBA"):
                    continue
                if image.size[0] < min_size[0] or image.size[1] < min_size[1]:
                    continue

                # Guardar imagen
                ruta = os.path.join(carpeta_destino, f"{objeto.replace(' ', '_')}_{descargadas+1}.jpg")
                image.convert("RGB").save(ruta, "JPEG")
                descargadas += 1
            except Exception:
                continue

    print(f"\n✅ Dataset listo: {descargadas} imágenes de '{objeto}' guardadas en {carpeta_destino}")

# Ejemplo
descargar_dataset("good conductive bus bar", carpeta_destino="dataset_good conductive bus bar", cantidad=500)
