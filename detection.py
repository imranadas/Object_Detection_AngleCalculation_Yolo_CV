def detection_tensors(src):
    from ultralytics import YOLO
    model = YOLO("#model") # Loading Trained Model
    results = model.predict(src, conf = 0.65) # Inference
    return results