import os
from datetime import datetime
import torch
import torch
from PIL import Image
from torch import optim
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import csv
from datetime import datetime
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F


'''
This class(Mydataset_png) load the training and testing data;
the data format requires png or jpg;
the training data will be preproceeded:pad+crop+flip;
the testing  data will not be preproceeded;

the return data format is torch.tensor(torch.FloatTensor)
'''


class Mydataset_png_client(Dataset):
    def __init__(self,mode,dataset,datalist):
        self.png_label_list=datalist

        #mode= "train" or "test"
        self.mode = mode
        #dataset:"your dataset folder name "
        self.dataset = dataset

        



            
            
    def __len__(self):
        if(self.mode=="train"):
            return len(self.png_label_list)
        if (self.mode == "test"):
            return len(self.png_label_list)

    def __getitem__(self, index):
        if(self.mode=="train"):
            #load png or jpg data
            #print(self.png_label_list[index])
            image = Image.open(os.path.join(self.dataset+"/train",self.png_label_list[index][0])).convert('RGB')
            #tranform into numpy format
            image_numpy =np.array(image,dtype="float32")
            #adjust format from 32_32_3 to 3_32_32 which is required by pytorch
            image_numpy=image_numpy.transpose(2,0,1)
            #padding + crop  : 3*32*32 -> 3*40*40   ->3*32*32
            featuremap=np.pad(image_numpy,((0,0),(4,4),(4,4)),'constant', constant_values=0) 
            #a random number in range[0,8] 
            x=np.random.randint(0, 9)
            y=np.random.randint(0, 9)
            w=32
            h=32
            featuremap=featuremap[:,x:x+w,y:y+w]
            #flip by chance of 0.5
            if torch.rand(1)<0.5:
                featuremap=np.flip(featuremap, 2)
                
                
            new_featuremap=featuremap.copy()
            featuremap_tensor=torch.from_numpy(new_featuremap)
            featuremap_tensor=featuremap_tensor.type(torch.FloatTensor)
            return featuremap_tensor,torch.tensor(self.png_label_list[index][1])

        
        
        if(self.mode=="test"):
            #load png or jpg data
            image = Image.open(os.path.join(self.dataset+"/test", self.png_label_list[index][0])).convert('RGB')
            image_numpy =np.array(image,dtype="float32")
            image_numpy=image_numpy.transpose(2,0,1)
            new_featuremap=image_numpy.copy()
            featuremap_tensor=torch.from_numpy(new_featuremap)
            featuremap_tensor=featuremap_tensor.type(torch.FloatTensor)
            return featuremap_tensor, torch.tensor(self.png_label_list[index][1])