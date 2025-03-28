#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 21:05:16 2025

@author: vishaladithya
"""
import timm
from torch import nn
from torch.utils.data import Dataset

#----------------------------------[Dataset]----------------------------------#

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

#----------------------------------[Models]----------------------------------#


class RoadSegmentationModel(nn.Module):
    
    def __init__(self):
        super(RoadSegmentationModel,self).__init__()
        
        self.base_model = timm.create_model('resnet50', pretrained=True,
                                            features_only = True)
#        self.encoder = nn.Sequential(*list(self.base_model.children())[:-2])
        
        self.upscale1 = nn.Sequential(
            nn.ConvTranspose2d(2048 , 1024, kernel_size=4,stride = 2,padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True))
        
        self.upscale2 = nn.Sequential(
            nn.ConvTranspose2d(1024 , 512, kernel_size=4,stride = 2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True))
        
        self.upscale3 = nn.Sequential(
            nn.ConvTranspose2d(512 , 256, kernel_size=4,stride = 2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True))
        
        self.upscale4 = nn.Sequential(
            nn.ConvTranspose2d(256 , 128, kernel_size=4,stride = 2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True))
        
        self.upscale5 = nn.ConvTranspose2d(128 , 1, kernel_size=4,stride = 2,padding=1)
    
        
    def forward(self,x):
        x = self.base_model(x)[-1]
        x = self.upscale1(x)        
        x = self.upscale2(x)      
        x = self.upscale3(x)      
        x = self.upscale4(x)
        x = self.upscale5(x)
        
        return x
    
