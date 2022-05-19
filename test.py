from multiprocessing.connection import wait
from utils import parse_configuration
from models.LoclModel import SegModel
from datasets import create_dataset, create_transformer
import torch
import matplotlib.pyplot as plt
import cv2

config = parse_configuration("config.json")
number_of_epoch = config['train_params']['number_of_epoch']
transformer = create_transformer(config['transformer_params'])
models_path = config['model_params']['save_folder']
print('initializing Dataset')
train,val = create_dataset(transformer,**config["train_dataset_params"])
print(number_of_epoch)
print('initializing fsModel')
model = SegModel(config['model_params'])
model.load_models(models_path,map_location=torch.device('cpu'))

c=0
for v in val:
    c+=1
    model.set_input(v)
    model.test()
    if c>2:
        break

print(model.accu())