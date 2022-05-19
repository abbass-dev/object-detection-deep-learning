import torch
import torchvision.transforms.functional as TF

class Resize(object):
    def __init__(self,output_size):
        self.output_size=output_size
    def __call__(self,sample):
        image = sample['image']
        label = sample['label']
        x,y = label
        #print(image.shape)
        _,o_w,o_h = image.shape
        t_w,t_h = self.output_size
        n_img = TF.resize(image,self.output_size)
        n_labels = [x*(t_w/o_w),y*(t_h/o_h)]
        return {'image':n_img,'label':n_labels}
'''
function to horizontaly flip image and labels
inputs: image,labels(x,y)
outputs: flipped image and labels
'''
class Horizontal_flip(object):
    def __init__(self,prob):
        self.prob = prob
    def __call__(self,sample):
        if torch.rand(1)>self.prob:
            return sample
        image = sample['image']
        label = sample['label']
        _,w,h = image.shape
        x,y = label
        n_img = TF.hflip(image)
        n_labels = [w-x,y]
        return {'image':n_img,'label':n_labels}
'''
function to verticaly flip image and labels
inputs: image,labels(x,y)
outputs: flipped image and labels
'''
class Vertical_flip(object):
    def __init__(self,prob):
        self.prob = prob
    def __call__(self,sample):
        if torch.rand(1)>self.prob:
            return sample
        image = sample['image']
        label = sample['label']
        _,w,h = image.shape
        x,y = label
        n_img = TF.vflip(image)
        n_labels = [x,w-y]
        return {'image':n_img,'label':n_labels}

'''
function to translate image and labels
inputs: image,labels(x,y),max translation ratio as tuple
outputs: translated image and labels
'''
class Translate(object):
    def __init__(self,prob,max_translation=(1,1)):
        self.prob = prob
        self.max_translation = max_translation
    def __call__(self,sample):
        if torch.rand(1)>self.prob:
            return sample
        image = sample['image']
        label = sample['label']
        _,w,h = image.shape
        x,y = label
        max_t_w,max_t_h = self.max_translation
        trans_coff_x= torch.rand(1)*2-1 #generate random number 0-1
        trans_coff_y= torch.rand(1)*2-1 
        x_t = int(trans_coff_x*max_t_h*h) #translation in x direction 0->max_translation*
        y_t = int(trans_coff_y*max_t_w*w)
        n_img = TF.affine(image,translate=(x_t,y_t),angle=0,shear=0,scale=1)
        n_labels = [x+x_t,y+y_t]
        return {'image':n_img,'label':n_labels}




class ScaleLabel(object):
    def __init__(self,image_size):
        self.image_size = image_size
    def __call__(self,sample):
        image = sample['image']
        label = sample['label']
        n_label = [ai/bi for ai,bi in zip(label,self.image_size)]
        return {'image':image,'label':n_label}
