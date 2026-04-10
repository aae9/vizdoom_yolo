import os
from ultralytics import YOLO

# Fix dataset directory
#os.environ["YOLO_DATASETS_DIR"] = r"C:\Users\Joshu\OneDrive\Desktop\Python\DoomEnvironment\doom\Lib\site-packages\ultralytics\cfg\datasets"

model = YOLO("yolo11n.pt")

model.train(
    data=r"doom.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolo11n-test",
    device="cpu"
)