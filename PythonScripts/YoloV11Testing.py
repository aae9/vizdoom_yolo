import os
from ultralytics import YOLO
import random

_script_dir = os.path.dirname(os.path.abspath(__file__))
test_image = random.choice(os.listdir(os.path.join(_script_dir, "../DoomDataset/model_data/images/val/")))
test_img_path = os.path.join(_script_dir, f"../DoomDataset/model_data/images/val/{test_image}")

# weights from folder
model_path = os.path.join(_script_dir, "../DoomDataset/model_weights/newtrainedyolo.pt")

model = YOLO(model_path)

model.predict(test_img_path, device="cpu", conf=0.5, save=True, save_dir = os.path.join(_script_dir,"../DoomDataset/model_predictions"))
