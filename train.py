from ultralytics import YOLO #imports YOLO Library

if __name__ == '__main__':
    model = YOLO('yolov8m.pt') #Loads yolov8m pretrained model
    results = model.train(data='dataset\data.yaml', epochs=100, imgsz=640) #training custom model for 100epochs