from ultralytics import YOLO
import numpy as np
import wandb

MODE = 'aug'
BATCH_SIZE = 32
PROJECT_DIR = '/Users/alliewandling/Documents/uva_masters/ML3/constellation-dataset/runs'
DEVICE = 'cpu'

models_to_train = [
    {
        'dataset': '/Users/alliewandling/Documents/uva_masters/ML3/constellation-dataset/configs/constellation.yaml',
        'model': './pretrained/yolov8x_visdrone.pt',
        'epochs': 1,
        'run_name': 'yolov8x_constellation_aug'
    }
]

# Load a model
N_WORKERS = 8
for model_to_train in models_to_train:
    model = YOLO(model_to_train['model'])  # build a new model from YAML
    config = {
            "EPOCHS": model_to_train['epochs'],
            "BATCH": BATCH_SIZE
        }
    with wandb.init(project='constellation', config=config, name=model_to_train['run_name']) as run:
        results = model.train(data=model_to_train['dataset'], 
                              epochs=model_to_train['epochs'], 
                              imgsz=832,
                              batch=BATCH_SIZE, 
                              save_period=1,
                              patience=0,
                              project = PROJECT_DIR,
                              pretrained=(model_to_train['pretrained'] if 'pretrained' in model_to_train else True),
                              cache=True, name = model_to_train['run_name'], device=DEVICE)