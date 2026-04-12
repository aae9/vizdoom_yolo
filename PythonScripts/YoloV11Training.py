import os
from ultralytics import YOLO

#model to use
model = YOLO("yolo11n.pt")

#yaml to use

yaml = r"../DoomDataset/model_data/yaml/doom.yaml"
model.train(
    data=yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolo11n-doom",
    device="cpu",
    save_dir= "../DoomDataset/model_weights"
)