import os
from ultralytics import YOLO

os.environ["YOLO_AUTOINSTALL"] = "False"
os.environ["ULTRALYTICS_SETTINGS"] = "False"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    #model to use
    model = YOLO(os.path.join(_script_dir, "../../DoomDataset/model_weights/yolo11s.pt"))

    #yaml to use

    yaml = os.path.join(_script_dir, "../../DoomDataset/model_data/yaml/doom.yaml")
    model.train(
        data=yaml,
        epochs=100,
        imgsz=320,
        batch=16,
        name="yolo11s-doom",
        device=0,
        patience=20,
        save_dir= os.path.join(_script_dir, "../../DoomDataset/model_weights")
    )
if __name__ == "__main__":    main() 