def detection_tensors(src):     #Detection function
    from ultralytics import YOLO    #importing library
    model = YOLO("book_yolov8.pt") # Loading Trained Model
    results = model.predict(src, conf = 0.5) # Inference
    draw_bounding_boxes(results)    
    return results

def draw_bounding_boxes(results):   #Bounding Boxes function and Logic to create a temporary Image with bounded boxes
    from PIL import Image
    import os 
    current_working_directory = os.getcwd()
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save("temp.jpg")