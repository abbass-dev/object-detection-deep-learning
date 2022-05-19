import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from os import path
from PIL import Image
from torchvision.transforms import ToTensor
'''
Custom pytorch dataset
Reads excel file and images from root_dir
args:
    root_dir
    transformer
return:
    torch.utils.data.Dataset
'''
class AmdDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir,transform=None):
        print(transform)
        self.images_path = []
        self.transform=transform
        excel_file = pd.read_excel(path.join(root_dir,"Fovea_location.xlsx"))
        images = excel_file['imgName'].values
        self.labels = excel_file[['Fovea_X','Fovea_Y']].values 
        for imageName in images:
            prefix = "AMD" if imageName[0] =='A' else "Non-AMD"
            image_path = path.join(root_dir,prefix,imageName)
            self.images_path.append(image_path)
    def __len__(self):
        return len(self.images_path)
    def __getitem__(self,idx):  
        image = ToTensor()(Image.open(self.images_path[idx]))
        label = self.labels[idx]
        if self.transform is not None:
            sample = self.transform({'image':image,'label':label})
            image = sample['image']
            label = sample['label']
        label = torch.tensor(label).double()
        image = image.double()
        return image,label