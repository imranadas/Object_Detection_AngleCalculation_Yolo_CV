from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.yaml')
    results = model.train(data='dataset\data.yaml', epochs=200, imgsz=640)