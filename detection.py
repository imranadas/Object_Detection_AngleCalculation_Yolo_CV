def detection_tensors(src):
    from ultralytics import YOLO
    model = YOLO("book_yolov8.pt") # Loading Trained Model
    results = model.predict(src, conf = 0.65) # Inference
    return results