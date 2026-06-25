import os
from ultralytics import YOLO

def train_new_model():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    #model to use
    model = YOLO(os.path.join(_script_dir, "../../DoomDataset/model_weights/yolo11s.pt")) 

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

def train_existing_model():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    #model to use
    model = YOLO(os.path.join(_script_dir, "../../DoomDataset/model_weights/weights/best.pt")) 

    #yaml to use
    yaml = os.path.join(_script_dir, "../../DoomDataset/model_data/yaml/doom.yaml")
    model.train(
        data=yaml,
        epochs=10,
        imgsz=320,
        batch=16,
        name="yolo11s-doom",
        device=0,
        patience=5,
        save=True,
        save_period = 3,
        save_dir= os.path.join(_script_dir, "../../DoomDataset/model_weights/finetuned"),
    )

if __name__ == "__main__":
    #train_new_model()
    train_existing_model()