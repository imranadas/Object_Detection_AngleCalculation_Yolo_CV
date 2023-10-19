import os   #Importing OS Library for File and Folder Manulpulation
import shutil   #Complementary Library to OS
from angle_calc import calculatation    #Importing Function to be used for Angle Calculation
import cv2  #openCV2 library for pre and post image processing

input_folder = input("Enter Path to Images: ")  #asking user for PATH to mage folder for detection-calculation

output_folder = os.getcwd() #retrieving current working directory
output_folder = os.path.join(output_folder, "results")  #generating path for results folder in CWD
if os.path.exists(output_folder):   #logic for checking and creation of results folder, Pre-Existing Folder and contents will be Deleted
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

def show_image(image):  #function to show image
    cv2.imshow("OutPut" ,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for image_name in os.listdir(input_folder): #passing path of image folder for further processing
    if image_name.endswith('.jpg'): #checking format of the image file
        processed_image = calculatation(os.path.join(input_folder, image_name)) #calling and storing calculation function and its output on the image passed to it
        new_image_name = image_name.replace('.jpg', '_calculated_.jpg') #generating finename for processed image
        output_path = os.path.join(output_folder, new_image_name)   #generating path for process image to be stored
        cv2.imwrite(output_path, processed_image)   #saving processed image locally
        print(f"Processed image saved as: {output_path}")
        show_image(processed_image)     #showing output to user

os.remove("temp.jpg") #removal of temp.jpg file used during the routine