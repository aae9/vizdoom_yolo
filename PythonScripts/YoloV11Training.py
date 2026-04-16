import os
from ultralytics import YOLO

_script_dir = os.path.dirname(os.path.abspath(__file__))
#model to use
model = YOLO("yolo11n.pt")

#yaml to use

yaml = os.path.join(_script_dir, "../DoomDataset/model_data/yaml/doom.yaml")
model.train(
    data=yaml,
    epochs=50,
    imgsz=320,
    batch=18,
    name="yolo11n-doom",
    device="cpu",
    save_dir= os.path.join(_script_dir, "../DoomDataset/model_weights")
)