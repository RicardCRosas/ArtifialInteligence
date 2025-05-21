from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # === CONFIGURACIÓN ===
    model_path = "runs/segment/train9/weights/best.pt"
    val_images_dir = r"C:\Users\Ricardo\Downloads\Ricardo Carballido TRain.v2i.yolov11\valid\images"
    results_folder = "runs/segment/train9"
    output_report = os.path.join(results_folder, "yolo11_analysis_report.txt")

    # === CARGAR MODELO ===
    model = YOLO(model_path)
    print("✅ Modelo cargado desde:", model_path)
    model.info()

    # === VALIDACIÓN ===
    print("\n📊 Ejecutando validación sobre conjunto de validación...")
    metrics = model.val()

    # === LEER PÉRDIDAS DE results.csv ===
    metrics_path = os.path.join(results_folder, "results.csv")
    box_loss = seg_loss = None
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        box_loss = df["train/box_loss"].iloc[-1]
        seg_loss = df["train/seg_loss"].iloc[-1]

    # === GUARDAR REPORTE ===
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("🧠 INFORME COMPLETO DE ENTRENAMIENTO - YOLOv11 SEGMENTACIÓN\n")
        f.write("===========================================================\n")
        f.write(f"📁 Modelo: {model_path}\n\n")

        f.write("📈 MÉTRICAS GENERALES (detección)\n")
        f.write(f"- Precisión media:        {metrics.box.mp:.4f}\n")
        f.write(f"- Recall medio:           {metrics.box.mr:.4f}\n")
        f.write(f"- mAP@0.5:                {metrics.box.map50:.4f}\n")
        f.write(f"- mAP@0.5:0.95:           {metrics.box.map:.4f}\n")
        if box_loss is not None:
            f.write(f"- Último box loss:        {box_loss:.4f}\n")

        f.write("\n📈 MÉTRICAS GENERALES (segmentación)\n")
        f.write(f"- Precisión media (mask): {metrics.seg.mp:.4f}\n")
        f.write(f"- Recall medio (mask):    {metrics.seg.mr:.4f}\n")
        f.write(f"- mAP@0.5 (mask):          {metrics.seg.map50:.4f}\n")
        f.write(f"- mAP@0.5:0.95 (mask):     {metrics.seg.map:.4f}\n")
        if seg_loss is not None:
            f.write(f"- Último seg loss:        {seg_loss:.4f}\n")

        f.write("\n📊 DETALLES DEL MODELO\n")
        f.write(f"- Parámetros totales: {sum(p.numel() for p in model.model.parameters()):,}\n")
        f.write(f"- Capas: {len(list(model.model.modules()))}\n")
        f.write(f"- GFLOPs aproximados: 10.4\n\n")

        # === AP por clase ===
        box_ap50 = metrics.box.ap50
        box_ap95 = metrics.box.ap
        seg_ap50 = metrics.seg.ap50
        seg_ap95 = metrics.seg.ap

        f.write("🔍 RESULTADOS POR CLASE:\n")
        for i, name in enumerate(metrics.names):
            f.write(f"Clase: {name}\n")
            f.write(f"  - Box(P):     {metrics.box.p[i]:.3f}\n")
            f.write(f"  - Box(R):     {metrics.box.r[i]:.3f}\n")
            f.write(f"  - Box(mAP50): {box_ap50[i]:.3f}\n")
            f.write(f"  - Box(mAP95): {box_ap95[i]:.3f}\n")
            f.write(f"  - Mask(P):    {metrics.seg.p[i]:.3f}\n")
            f.write(f"  - Mask(R):    {metrics.seg.r[i]:.3f}\n")
            f.write(f"  - Mask(mAP50):{seg_ap50[i]:.3f}\n")
            f.write(f"  - Mask(mAP95):{seg_ap95[i]:.3f}\n\n")

    print(f"\n✅ Informe guardado en: {output_report}")

    # === GRÁFICAS DE ENTRENAMIENTO (con validación de columnas) ===
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        plt.figure(figsize=(14, 8))

        if "train/box_loss" in df and "train/seg_loss" in df:
            plt.subplot(2, 2, 1)
            plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")
            plt.plot(df["epoch"], df["train/seg_loss"], label="Seg Loss")
            plt.title("Pérdida durante entrenamiento")
            plt.xlabel("Época")
            plt.ylabel("Loss")
            plt.legend()

        if "metrics/precision(B)" in df and "metrics/recall(B)" in df:
            plt.subplot(2, 2, 2)
            plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precisión")
            plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
            plt.title("Precisión y Recall")
            plt.xlabel("Época")
            plt.legend()

        # Compatibilidad para nombres alternativos de mAP
        mAP_50_key = "metrics/mAP_0.5(B)" if "metrics/mAP_0.5(B)" in df else "metrics/mAP50"
        mAP_95_key = "metrics/mAP_0.5:0.95(B)" if "metrics/mAP_0.5:0.95(B)" in df else "metrics/mAP50-95"

        if mAP_50_key in df and mAP_95_key in df:
            plt.subplot(2, 2, 3)
            plt.plot(df["epoch"], df[mAP_50_key], label="mAP@0.5")
            plt.plot(df["epoch"], df[mAP_95_key], label="mAP@0.5:0.95")
            plt.title("mAP por época")
            plt.xlabel("Época")
            plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No se encontró 'results.csv'. ¿El entrenamiento fue interrumpido?")


if __name__ == "__main__":
    main()
