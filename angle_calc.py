from ultralytics import YOLO
import cv2
from detection import detection_tensors
import numpy as np

src = cv2.imread("test.jpg")
detections = detection_tensors(src)

for detection in detections:
    data = detection.boxes.data.clone().detach()
    for subimage_parameters in data:
        x1, y1, x2, y2, score, label = subimage_parameters
        x1 = int(x1.item())
        y1 = int(y1.item())
        x2 = int(x2.item())
        y2 = int(y2.item())
        subimage = src[y1:y2, x1:x2]
        gauss_blur = cv2.GaussianBlur(subimage,(3,3), cv2.BORDER_DEFAULT)
        grayscale_subimage = cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(grayscale_subimage, 50, 200, L2gradient=True)
        cv2.imshow("Canny_Edges", canny_edges)
        cv2.imshow("SubImage", subimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()