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
        print (x1, x2, y1, y2)
    '''
        subimage = src[y1:y2, x1:x2]
        grayscale_subimage = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(grayscale_subimage, 50, 150)
        median_blurred_edges = cv2.medianBlur(canny_edges, 3)
        thresholded_edges = cv2.threshold(median_blurred_edges, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        mask = np.zeros(thresholded_edges.shape, dtype=np.uint8)
        mask[y1 - y1:y2 - y1, x1 - x1:x2 - x1] = 255
        masked_edges = cv2.bitwise_and(thresholded_edges, mask)
        masked_edges = cv2.resize(masked_edges, (subimage.shape[1], subimage.shape[0]))
        highlighted_image = cv2.addWeighted(subimage, 1.0, cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.imshow('Image with Edges', highlighted_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''