import os 
import numpy as np
from PIL import Image
import random
import time
import copy
import pandas as pd
from matplotlib import pyplot as plt
import glob
import multiprocessing
from matplotlib import pyplot
import pathlib
import random as rd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorboardX import SummaryWriter


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models,transforms,utils
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
rd.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = 2





root = '/ssd_scratch/cvit/ashishmenon/ara_expts/'        




T = {'train': transforms.Compose([ transforms.Resize(224),
                                   transforms.RandomHorizontalFlip(p=0.8),
                                   transforms.RandomVerticalFlip(p=0.8),
                                   transforms.RandomRotation(90),
                                   transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))       
                                 ]) ,
      
    'val': transforms.Compose([  transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))       
                                 ]) ,
      
    'test': transforms.Compose([   transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))       
                                 ]) 
    }

datasets={x: dset.ImageFolder( os.path.join(root,x), T[x] ) for x in os.listdir(root) }

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'] , shuffle=True, batch_size=32 , num_workers=8)
dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'] , shuffle=True, batch_size=128 , num_workers=16)
dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'] , shuffle=False, batch_size=128 , num_workers=16)

device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
class_names=datasets['train'].classes

print(len(datasets['val']))

lst=os.listdir(root)

acc_list_train=[]
loss_list_train=[]

acc_list_val=[]
loss_list_val=[]

epoch_list=[]
epoch_train = []
epoch_val = []



def train_model(model, criterion, optimizer, scheduler_train , scheduler_val, num_epochs=10):
    since=time.time()
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc = 0.0
    thresh = 1000
    for epoch in range(num_epochs):
        epoch_loss={}
        epoch_acc = {}
        for phase in ['train','val']:
            print(phase,flush=True)
            if phase=='train':
                model.train()
            else:
                model.eval()
                
            running_loss=0
            running_corrects=0
            
            for images , labels in dataloaders[phase]:
                images=images.to(device)
                labels=labels.to(device)
                optimizer.zero_grad()        
                with torch.set_grad_enabled(phase=='train'):
    
                    outputs=model(images)                    
                    _,predictions=torch.max(outputs,1)
                    loss=criterion(outputs,labels)
                    
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                running_loss+=loss.item()*images.size(0)
                running_corrects+=torch.sum(predictions == labels.data)
                
            epoch_loss[phase]=running_loss/len(datasets[phase])
            epoch_acc[phase]=running_corrects.double()/len(datasets[phase])      
                
            if phase=='val':
                scheduler_train.step(epoch_loss[phase])
                scheduler_val.step(epoch_acc[phase])
        
            
            print('{} Loss:{:.4f} Acc:{:.4f} '.format(phase, epoch_loss[phase],epoch_acc[phase]),flush=True) 
            if phase == 'val' and epoch_acc[phase]>best_acc:
               
                best_acc=epoch_acc[phase]
                best_model_wts=copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),'/home/ashishmenon/ARA/Pytorch_imp/ara_pretrained/model_op/squeezenet_pretrained_acc.pth')
        
        writer.add_scalars('Loss',{'train':epoch_loss['train'],'val':epoch_loss['val']}, epoch)
        writer.close()
        writer.add_scalars('Acc',{'train':epoch_acc['train'],'val':epoch_acc['val']}, epoch)
        writer.close()

            
            
    model.load_state_dict(best_model_wts)   
    

    return model


num_ftrs = 1000
model_ft = nn.Sequential(models.squeezenet1_1(pretrained=True), nn.Linear(num_ftrs,8))


model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()


optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.00005, weight_decay=0.005)
exp_lr_scheduler_val = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft,verbose=True,mode='max',patience=5)
exp_lr_scheduler_train = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft,verbose=True,mode='min',patience=5)


writer = SummaryWriter(log_dir='./logs/pretrained_squeezenet/')


model_best = train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler_train,exp_lr_scheduler_val, num_epochs=200)


