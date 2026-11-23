from ultralytics import YOLO

# Load the custom VisDrone model
model = YOLO("pretrained/yolov8x_visdrone.pt")

# View model details (layers, parameters, class names)
model.info()
print(model.info(detailed=True))  # Detailed model information
#print(model.names)  # Shows VisDrone classes like pedestrian, car, van, etc.


