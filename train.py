from sklearn.metrics import accuracy_score
from utils import parse_configuration
from models.LoclModel import LoclModel
from datasets import create_dataset, create_transformer
from matplotlib import pyplot as plt

config = parse_configuration("config.json")
number_of_epoch = config['train_params']['number_of_epoch']
transformer = create_transformer(**config['transformer_params'])
print('initializing Dataset')
train,val = create_dataset(transformer,**config["train_dataset_params"])
print(number_of_epoch)
print('initializing fsModel')
model = LoclModel(config['model_params'])
data_size = len(train.dataset)
print('start training')
for epoch in range(number_of_epoch):
    running_loss = 0
    for data in train:
        model.set_input(data)
        model.forward()
        model.backward()
        model.optimize_parameters()
        running_loss += model.get_current_losses()['smothl1'].item()
    print(running_loss/data_size)
model.post_epoch_callback()
loss_histoy , accuracy_history = model.get_training_history()

plt.plot(loss_histoy)
plt.plot(accuracy_history)
plt.title('training history')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['loss', 'accu'], loc='upper left')
plt.show()