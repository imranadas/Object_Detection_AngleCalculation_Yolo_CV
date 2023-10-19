def detection_tensors(src):
    from ultralytics import YOLO
    model = YOLO("book_yolov8.pt") # Loading Trained Model
    results = model.predict(src, conf = 0.5) # Inference
    return results

def draw_bounding_boxes(results):
    from PIL import Image
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.show()
        im.save('results.jpg') 