from ultralytics import YOLO
from PIL import Image
import cv2
from detection import detection_tensors
import torch

src = "test.jpg"
detections = detection_tensors(src)
for detection in detections:
    data = detection.boxes.data.clone().detach()
    for subimage_parameters in data:
        x1, y1, x2, y2, score, label = subimage_parameters
        x1 = x1.item()
        y1 = y1.item()
        x2 = x2.item()
        y2 = y2.item()
        subimage = src[y1:y2, x1:x2]
        edges = cv2.Canny(subimage, threshold1=30, threshold2=100)
        image_with_edges = cv2.bitwise_and(src, src, mask=edges)

cv2.imshow('Image with Edges', image_with_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()