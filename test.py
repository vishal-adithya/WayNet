"""
Created on Fri Mar 28 20:33:51 2025

@author: vishaladithya
"""

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
    
    
model = RoadSegmentationModel()
model.load_state_dict(torch.load("rs1(high).pth",map_location=torch.device("mps")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()])

def Preprocess(x):
    x = Image.open(x).convert("RGB")
    x = transform(x)
    x = x.unsqueeze(0)
    return x

img = "test_img2.jpg"
img = Preprocess(img)
device = torch.device(("mps"))
model.to(device)
img = img.to(device)

with torch.no_grad():
    out = model(img)
yhat = out.squeeze().cpu().numpy()
threshold = 0.5  # Adjust as needed
binary_mask = (yhat > threshold).astype(np.uint8) * 255
y = cv2.imread("test_img.jpg",cv2.IMREAD_COLOR_RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(y, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(binary_mask, cmap="gray")

plt.show()

def overlay_mask(image_path, mask, alpha=0.5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))


    color_mask = np.zeros_like(image)
    color_mask[:, :, 2] = mask * 255
    blended = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    return blended


segmentation_overlay = overlay_mask("test_img2.jpg", yhat)


plt.figure(figsize=(10, 5))
plt.imshow(segmentation_overlay)
plt.axis("off")
plt.title("Segmentation Overlay")
plt.show()