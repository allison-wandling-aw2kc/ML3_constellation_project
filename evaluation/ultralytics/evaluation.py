from ultralytics import YOLO
import os
folder = '/Users/alliewandling/Documents/uva_masters/ML3/constellation-dataset/pretrained/' # Change to the model path (can be downloaded from models/)
for file in os.listdir(folder):
    print(file)
    model = YOLO(folder+file)
    model.val(data='/Users/alliewandling/Documents/uva_masters/ML3/constellation-dataset/configs/constellation.yaml')
