
from utils import parse_configuration
from models.LoclModel import LoclModel
from datasets import create_dataset, create_transformer

config = parse_configuration("config.json")
number_of_epoch = config['train_params']['number_of_epoch']
transformer = create_transformer(**config['transformer_params'])
print('initializing Dataset')
_,val = create_dataset(transformer,**config["train_dataset_params"])
print(number_of_epoch)
print('initializing fsModel')
model = LoclModel(config['model_params'])
print('start training')

for data in val:
    model.set_input(data)
    model.test()
accuracy = model.accuracy()
print(f'model accuracy on test test {accuracy}')

