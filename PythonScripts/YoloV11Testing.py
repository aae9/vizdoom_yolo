import os
from ultralytics import YOLO

test_image = r"../DoomDataset/test_data/Barons_of_Hell.png"

model_path = r"../DoomDataset/model_weights/"

model = YOLO(model_path + "yolo11n.pt")

model.predict(test_image, device="cpu", conf=0.5, save=True, save_dir = "../DoomDataset/model_predictions")
