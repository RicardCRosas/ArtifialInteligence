import torch
from ultralytics import YOLO

def main():
    # Verificación de versiones
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        # Intentar cargar el modelo YOLOv11
        model = YOLO("yolo11n-seg.pt")  # Asegúrate de tener este archivo en el mismo folder
        print("Modelo cargado exitosamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")

    # Entrenar el modelo con tu dataset personalizado
    results = model.train(
        data=r"C:\Users\Ricardo\Downloads\Ricardo Carballido TRain.v3i.yolov11\data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        device=0  # Usando CPU para verificación inicial
    )

if __name__ == '__main__':
    main()