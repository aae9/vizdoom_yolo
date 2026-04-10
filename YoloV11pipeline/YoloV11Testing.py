import os
from ultralytics import YOLO


model = YOLO("best.pt")

model.predict("Please.jpg", device="cpu", conf=0.5, save=True)
