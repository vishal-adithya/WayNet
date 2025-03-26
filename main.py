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
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from torch import nn
import albumentations as alb
from PIL import Image
from torchvision import transforms

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
x = [print(i) for i in train_dataset[0]]


i,m = train_dataset[0]
print(f"Image Shape: {i.shape}")
print(f"Mask Shape: {m.shape}")

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

for i,m in train_loader:
    break

print(f"Batched Image Shape: {i.shape}")
print(f"Batched Mask Shape: {m.shape}")