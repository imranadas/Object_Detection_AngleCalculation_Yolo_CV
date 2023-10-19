from ultralytics import YOLO
import cv2
from detection import detection_tensors
import numpy as np
from scipy import stats as st
import math

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
        gauss_blur = cv2.GaussianBlur(subimage,(1,1), cv2.BORDER_DEFAULT)
        grayscale_subimage = cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(grayscale_subimage, 0, 200, L2gradient=True)
        lines = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=1)
        line_image = np.zeros_like(canny_edges)
        book_slope = 0
        angles_array = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-5)
                if abs(slope) > (0.1) and abs(slope) < (2*np.pi - 0.1):
                    cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)
                    angle_degrees = np.arctan(slope) * 180 / np.pi
                    angles_array = np.append(angles_array, angle_degrees)
        book_slope = st.mode(angles_array)
        if math.isnan(book_slope[0]):
            print("Angle Not Calculated")
        else:
            print(f"Angle of inclination: {book_slope[0]} degrees")
        horizontal_lines = cv2.HoughLinesP(line_image, 1, np.pi / 180, threshold=50, minLineLength=25, maxLineGap=5)
        if horizontal_lines is not None:
            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 0, 1)
        vertical_lines = cv2.HoughLinesP(line_image, 1, np.pi / 180, threshold=50, minLineLength=25, maxLineGap=5)
        if vertical_lines is not None:
            for line in vertical_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 0, 1)
        cv2.imshow("Canny_Edges", canny_edges)
        cv2.imshow("Canny_Edges_Pruning", line_image)
        cv2.imshow("SubImage", subimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()