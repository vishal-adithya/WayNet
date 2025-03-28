"""
Created on Wed Mar 26 11:38:48 2025

@author: vishaladithya
"""

import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
import timm
import warnings
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from torch import nn
import albumentations as alb
from PIL import Image
from torchvision import transforms
from torchinfo import summary
from utils import *

torch.backends.mps.is_available()
device = torch.device("mps")

warnings.warn("ignore")


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


    
    
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


train_dataset = RoadSegmentationDF(data = df["train"],transform=transform)

for i,m in train_dataset:
    break

train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)

for i_,m_ in train_loader:
    break

    
model = RoadSegmentationModel().to(device)
summary(model,input_size = (16,3,256,256))

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
        
torch.save(model.state_dict(),"rs1(high).pth")  

  
    
    
    
    