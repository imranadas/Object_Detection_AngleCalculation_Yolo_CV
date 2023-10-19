import cv2  #importing openCV2 library for image pre and post processing
from detection import detection_tensors     #imporint objecting detecting function
import numpy as np  #importing numpy library for mathematical calculations
from scipy import stats as st   #importing stats module for required stats functions
import math #importing math library for required functions

def calculatation(path):    #calling calc function for processing iamges and angle calculation
    src = cv2.imread(path)  #reading image in cv2 object
    detections = detection_tensors(src)     #taking detection tensor objects in variable
    output_image = cv2.imread("temp.jpg")   #Bounding Box Images from Detection function stored temporarily
    for detection in detections:    #accessing BB for each Detected Book
        data = detection.boxes.data.clone().detach()    #tensors object with pixel coordinates
        i=0
        for subimage_parameters in data:
            x1, y1, x2, y2, score, label = subimage_parameters      #logic for pixel coordinates extraction from tensors
            x1 = int(x1.item())
            y1 = int(y1.item())
            x2 = int(x2.item())
            y2 = int(y2.item())

            subimage = src[y1:y2, x1:x2]    #extracting BB subimage
            gauss_blur = cv2.GaussianBlur(subimage,(1,1), cv2.BORDER_DEFAULT)   #Gaussian Blur over Subimage
            grayscale_subimage = cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2GRAY)   #GrayScale over Blurred Image
            canny_edges = cv2.Canny(grayscale_subimage, 0, 200, L2gradient=True)    #Detecting Edges via Canny Edge Detection Method
            lines = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=1)      #Extracting Lines from the Edges
            line_image = np.zeros_like(canny_edges)

            book_slope = 0      #Logic to Filtering Lines and Calculating inclination angles
            angles_array = []
            if lines is not None:
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    slope = (ly2 - ly1) / (lx2 - lx1 + 1e-5)
                    if abs(slope) > (0.1) and abs(slope) < (2*np.pi - 0.1):
                        cv2.line(line_image, (lx1, ly1), (lx2, ly2), 255, 1)
                        angle_degrees = np.arctan(slope) * 180 / np.pi
                        angles_array = np.append(angles_array, angle_degrees)
            horizontal_lines = cv2.HoughLinesP(line_image, 1, np.pi / 180, threshold=50, minLineLength=25, maxLineGap=5)
            if horizontal_lines is not None:    #Horizontal line Pruning
                for line in horizontal_lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    cv2.line(line_image, (lx1, ly1), (lx2, ly2), 0, 1)
            vertical_lines = cv2.HoughLinesP(line_image, 1, np.pi / 180, threshold=50, minLineLength=25, maxLineGap=5)
            if vertical_lines is not None:      #Vertical line Pruning
                for line in vertical_lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    cv2.line(line_image, (lx1, ly1), (lx2, ly2), 0, 1)

            book_slope = st.mode(angles_array)      #Logic to Draw Angle as text inside Bounding Boxes
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 0, 0)
            font_scale = 0.5
            thickness = 2
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2 + 25*i)
            i+=1
            if math.isnan(book_slope[0]):
                text = "NaN"
            else:
                text = f"{abs(int(book_slope[0]))} degrees"
                output_image = cv2.putText(output_image, text, (x,y),font, font_scale, color, thickness)
        return output_image