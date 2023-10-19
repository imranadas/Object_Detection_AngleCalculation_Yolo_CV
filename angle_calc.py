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
        canny_edges = cv2.Canny(grayscale_subimage, 0, 200, L2gradient=True)
        lines = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        line_image = np.zeros_like(canny_edges)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)
        cv2.imshow("Canny_Edges", canny_edges)
        cv2.imshow("Canny_Edges_Pruning", line_image)
        cv2.imshow("SubImage", subimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()