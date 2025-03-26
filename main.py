"""
Created on Wed Mar 26 11:38:48 2025

@author: vishaladithya
"""

import torch
torch.device("mps")

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import timm
import torch
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from torch import nn
import albumentations as alb
from PIL import Image
from torchvision import transforms
from torchinfo import summary
from torchvision.models.segmentation import deeplabv3_resnet50

torch.backends.mps.is_available()
device = torch.device("mps")


df= load_dataset("bnsapa/road-detection")
df_train = df["train"]

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.imshow(df_train["image"][0])
plt.subplot(1,3,2)
plt.imshow(df_train["segment"][0])
plt.subplot(1,3,3)
plt.imshow(df_train["lane"][0])
plt.show()

# Building the Dataloader and Batching the data

class RoadSegmentationDF(Dataset):
    
    def __init__(self,data,transform = None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        sample = self.data[idx]
        
        image = sample["image"].convert("RGB")
        mask = sample["segment"].convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image,mask
    
    
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


train_dataset = RoadSegmentationDF(data = df["train"],transform=transform)
for i,m in train_dataset:
    break

train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)
for i_,m_ in train_loader:
    break

# Building the Neural net

class RoadSegmentationModel(nn.Module):
    
    def __init__(self):
        super(RoadSegmentationModel,self).__init__()
        
        self.base_model = timm.create_model('resnet50', pretrained=True,
                                            features_only = True)
#        self.encoder = nn.Sequential(*list(self.base_model.children())[:-2])
        
        self.upscale = nn.ConvTranspose2d(2048, 1024, kernel_size=4,
                                          stride = 2,padding=1)
        self.upscale2 = nn.ConvTranspose2d(1024 , 512, kernel_size = 4,
                                           stride = 2,padding = 1)
        self.upscale3 = nn.ConvTranspose2d(512, 256, kernel_size = 4,
                                           stride = 2,padding = 1)
        self.upscale4 = nn.ConvTranspose2d(256, 128, kernel_size = 4,
                                           stride = 2,padding = 1)
        self.upscale5 = nn.ConvTranspose2d(128, 1, kernel_size = 4,
                                           stride = 2,padding = 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
 #      x = self.encoder(x)
        x = self.base_model(x)[-1]
        x = self.upscale(x)
        x = self.upscale2(x)
        x  =self.upscale3(x)
        x = self.upscale4(x)
        x = self.upscale5(x)
        x = self.sigmoid(x) 
        return x
    
model = RoadSegmentationModel().to(device)
summary(model,input_size = (8,3,256,256))

# Contructing the Training loop

EPOCHS = 10
opt = torch.optim.Adam(model.parameters(),lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for image,mask in tqdm(train_loader,desc = "Training Loop"):
        image,mask = image.to(device,dtype = torch.float32),mask.to(device,dtype = torch.float32)
        mask = mask.float()/255.0
        opt.zero_grad()
        yhat = model(image)
        loss = criterion(yhat,mask)
        loss.backward()
        opt.step()
        running_loss+=loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f"EPOCH: [{epoch+1}/{epoch}] , LOSS: [{train_loss:.4f}]")
        
torch.save(model.state_dict(),"rs1(low).pth")  

  
    
    
    
    