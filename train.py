from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')
    results = model.train(data='dataset\data.yaml', epochs=100, imgsz=640)