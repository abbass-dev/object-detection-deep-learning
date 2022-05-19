'''
U-net model
consists of conv layers
and residul connections 

imputs :params{
input_channels
initial_filter
number_out
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self,params={'input_channels':3,'initial_filter':16,'number_out':2}):
        super(UNet,self).__init__()
        C_in = params['input_channels']
        init_f = params['initial_filter']
        num_outputs = params['number_out']
        self.conv1 = nn.Conv2d(C_in,init_f,kernel_size=3,padding=1,stride=2)
        self.conv2 = nn.Conv2d(C_in+init_f,  2*init_f,kernel_size=3,padding=1,stride=1)
        self.conv3 = nn.Conv2d(C_in+init_f*3,4*init_f,kernel_size=3,padding=1,stride=1)
        self.conv4 = nn.Conv2d(C_in+init_f*7,8*init_f,kernel_size=3,padding=1,stride=1)
        self.conv5 = nn.Conv2d(C_in+init_f*15,16*init_f,kernel_size=3,padding=1,stride=1)
        self.fcl   = nn.Linear(16*init_f,num_outputs)

    def forward(self,x):
        identity = F.avg_pool2d(x,kernel_size = 4)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = torch.cat((x,identity),dim=1)
        
        identity = F.avg_pool2d(x,kernel_size = 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = torch.cat((x,identity),dim=1)
        
        identity = F.avg_pool2d(x,kernel_size = 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2)
        x = torch.cat((x,identity),dim=1)
        
        identity = F.avg_pool2d(x,kernel_size = 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x,2)
        x = torch.cat((x,identity),dim=1)
        
        x = F.relu(self.conv5(x))
        
        x = F.adaptive_avg_pool2d(x,1)
        x = torch.flatten(x,start_dim=1)
        x = self.fcl(x)
        return x
