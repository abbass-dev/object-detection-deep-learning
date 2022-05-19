from sklearn import datasets
from sklearn.utils import shuffle
import torch
from datasets.data import AmdDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset
from transformers import Resize,Horizontal_flip,Vertical_flip
from torchvision.transforms import Compose


def create_dataset(transformer=None,**params):
    dataset_path = params['dataset_path']
    dataset = AmdDataset(dataset_path,transformer)
    if params['val_ratio'] is not False:
        spliter = ShuffleSplit(n_splits=1,test_size=0.2)
        for train_idx,val_idx in spliter.split(range(len(dataset))):
            train_ds = Subset(dataset,indices=train_idx)
            val_ds = Subset(dataset,indices=val_idx)
        train_dl = DataLoader(train_ds,**params['loader_params'])
        val_dl = DataLoader(val_ds,**params['loader_params'])
        return train_dl,val_dl
    else:
        return DataLoader(dataset,**params['loader_params'])
    
def create_transformer(**params):
    h_w = params['height_width']
    h_p = params['vertical_flip_p']
    v_p = params["horizontal_flip_p"]
    transformer = Compose([
    Resize(h_w),
    Horizontal_flip(h_p),
    Vertical_flip(v_p)])
    return transformer
