from ultralytics import YOLO
model = YOLO('yolo11n.pt')
#model.train(data=f"C:\\Users\\Joshu\\OneDrive\\Desktop\\Python\\DoomEnvironment\\coco128.yaml" , epochs=5, imgsz=640, batch=16, device="cpu", lr0=0.01)
results = model.predict(source="bus.jpg", save=True, save_txt=True, conf=0.25)
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk