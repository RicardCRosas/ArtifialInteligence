import cv2
from ultralytics import YOLO

# Carga tu modelo entrenado (ajusta el path al tuyo)
model = YOLO("runs/segment/train10/weights/best.pt")  # o por ejemplo "yolov8n-seg.pt"

# Abre la cámara (índice 0 usualmente es la cámara integrada)
cap = cv2.VideoCapture(1)

# Verifica si la cámara se abrió correctamente
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("✅ Cámara activada. Presiona 'q' para salir.")

# Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame.")
        break

    # Realiza la inferencia
    results = model.predict(source=frame, show=False, stream=False, conf=0.3)

    # Dibuja las máscaras y bounding boxes
    annotated_frame = results[0].plot()

    # Muestra el frame anotado
    cv2.imshow("YOLO Segmentación - Webcam", annotated_frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
