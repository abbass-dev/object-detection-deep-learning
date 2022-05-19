from abc import ABC,abstractmethod
from typing import OrderedDict
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
"""Initialize the BaseModel class.
        Parameters:
            configuration: Configuration dictionary.
        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define these lists:
            -- self.network_names (str list):       define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
"""
class Base(ABC):
    def __init__(self):
        self.verbose = 1
        self.network_names = []
        self.loss_names = []
        self.optimizers = []
        self.schedulers =  []# added by subcalss
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def move_models_to_device(self):
        """save models to the disk"""
        for model_name in self.network_names:
            if isinstance(model_name,str):
                model = getattr(self,'net'+model_name)
                model = model.to(self.device)

    def set_input(self,input):
        #sets input and move data to appropriate device 
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device)

    def train(self):
        #sets models parameters as trainable
        for net in self.network_names:
            if isinstance(net,str):
                net = getattr(self,'net'+net)
                net.train()

    def test(self):
        #sets models parameters in testing mode
        for net in self.network_names:
            if isinstance(net,str):
                net = getattr(self,'net'+net)
                net.eval()

    @abstractmethod
    def forward(self):
        """run forwad pass on pre-set data in self.inputs ;called in every training iteration"""
        pass

    @abstractmethod
    def backward(self):
        """Calculates loss; called in every training iteration"""
        pass
    @abstractmethod
    def optimize_parameters(self):
      """update network weights; called in every training iteration"""
      pass


    def update_learning_rate(self):
        """update learning rate ; called after every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {0:.7f}'.format(lr))

    def get_current_losses(self):
        """returns current losses"""
        losses = OrderedDict()
        for name in self.loss_names:
            if isinstance(name,str):
                loss = getattr(self,'loss_'+name)
                losses[name] = loss
        return losses
    
    def save_models(self,path):
        """save models to the disk"""
        for model_name in self.network_names:
            if isinstance(model_name,str):
                model = getattr(self,'net'+model_name)
                path = self+"/"+model_name+".pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                },f=path)
                
    def load_models(self,path,map_location):
        """save models to the disk"""
        for model_name in self.network_names:
            if isinstance(model_name,str):
                model = getattr(self,'net'+model_name)
                checkpoint = torch.load(path+"/net"+model_name+".pt",map_location=map_location)
                model.load_state_dict(checkpoint['model_state_dict'],)





