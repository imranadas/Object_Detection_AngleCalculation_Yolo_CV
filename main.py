import os
from angle_calc import calculatation
import cv2

input_folder = "D:\\IITB\\Coding_Programming\\Python_YOLO_OpenCV_Book_Detection_Angle_Processing\\book_image"
output_folder = "D:\\IITB\\Coding_Programming\\Python_YOLO_OpenCV_Book_Detection_Angle_Processing\\result_book_image"
os.rmdir(output_folder)
os.mkdir(output_folder)

def show_image(image):
    cv2.imshow("OutPut" ,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for image_name in os.listdir(input_folder):
    if image_name.endswith('.jpg'):
        processed_image = calculatation(os.path.join(input_folder, image_name))
        new_image_name = image_name.replace('.jpg', '_calculated_.jpg')
        output_path = os.path.join(output_folder, new_image_name)
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image saved as: {output_path}")
        show_image(processed_image)

os.remove("temp.jpg")