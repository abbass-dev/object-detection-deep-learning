from abc import ABC,abstractmethod
from cProfile import label
import torch
from models.UNet import UNet
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from models.base import Base
import torch.nn.functional as F
from losses import diceLoss

class LoclModel(Base):
    def __init__(self,params):
        super(LoclModel,self).__init__()

        self.params =params
        self.network_names = []
        self.loss_names= []
        self.optimizers = []
        self.netU = UNet(params).to(self.device).double()
        self.network_names.append('U')
        self.opt = Adam(self.netU.parameters(),lr=0.001)
        self.optimizers.append(self.opt)
        self.schedulers.append(StepLR(self.opt,step_size=10, gamma=0.3))
        self.loss_history = 0
        # storing predictions and labels for validation
        self.val_predictions = []
        self.val_labels = []
        self.val_images = []
        self.accuracy_history = []
        self.best_accuracy = 0

    def forward(self):
        self.output = self.netU(self.input)
        return self.output

    def backward(self):
        self.loss_names.append("smothl1")
        self.loss_smothl1 = F.smooth_l1_loss(self.output,self.label,reduce='sum')

    def optimize_parameters(self):
        self.loss_smothl1.backward()
        self.opt.step()
        self.opt.zero_grad()
    
    def test(self):
        super().test()
        with torch.no_grad():
            self.forward()
        #print(self.input[0].shape)
        self.val_predictions.append(torch.sigmoid(self.output))
        self.val_images.append(self.input)
        self.val_labels.append(self.label)
    
    def accuracy(self):
        labels = torch.cat(self.val_labels,dim=0)
        target = torch.cat(self.val_predictions,dim=0)
        _,accuracy = diceLoss(labels,target)
        self.accuracy_history.append(accuracy)
        self.accuracy = accuracy
        #reset validation data
        self.val_predictions=0
        self.val_images=0
        self.val_labels=0
        return accuracy

    def post_epoch_callback(self):
       self.update_learning_rate()
       if self.accuracy>self.best_accuracy:
           self.best_accuracy = self.accuracy
           self.save_models(self.params['model_params']['"save_folder"'])

    def get_training_history(self):
        return self.loss_history , self.accuracy
