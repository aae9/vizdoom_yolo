import os
from ultralytics import YOLO


def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    #model to use
    model = YOLO(os.path.join(_script_dir, "../../DoomDataset/model_weights/yolo11s.pt"))  # ← change this to your desired model checkpoint"))

    #yaml to use

    yaml = os.path.join(_script_dir, "../../DoomDataset/model_data/yaml/doom.yaml")
    model.train(
        data=yaml,
        epochs=21,
        imgsz=320,
        batch=16,
        name="yolo11s-doom",
        device=0,
        patience=20,
        save=True,
        save_period = 3,
        save_dir= os.path.join(_script_dir, "../../DoomDataset/model_weights")
    )
if __name__ == "__main__":    main() 