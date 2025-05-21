from ultralytics.data.annotator import auto_annotate
import os
import cv2
from ultralytics import YOLO

auto_annotate(
    data= r"C:\Users\Ricardo\PycharmProjects\ArtifialInteligence\Limpieza y de duplicaccion IA\imagenes_redimensionadas2",
    det_model = YOLO("yolo11n-seg.pt"), # Asegúrate de tener este archivo en el mismo folder
    sam_model="sam_b.pt",     # asegúrate de tenerlo descargado
    imgsz=640,
    conf=0.25,
    iou=0.45,
    device="cuda"  # usa "cpu" si no tienes GPU
)
