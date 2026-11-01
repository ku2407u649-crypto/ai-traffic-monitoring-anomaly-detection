"""Fine-tune YOLO on a traffic dataset (UA-DETRAC, your own annotated videos, ...).

1. Put the dataset in YOLO format:   dataset/images/{train,val}  +  dataset/labels/{train,val}
2. Edit dataset.yaml (class names) if needed.
3. Run (GPU strongly recommended - Google Colab is fine):

     python train_yolo.py --data dataset.yaml --epochs 50

The best weights are saved to runs/detect/<name>/weights/best.pt. Put that path in
config.yaml -> model.weights.
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.yaml")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="traffic")
    a = ap.parse_args()
    model = YOLO(a.weights)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, name=a.name)
    m = model.val()
    print("mAP50:", m.box.map50, " mAP50-95:", m.box.map)


if __name__ == "__main__":
    main()
