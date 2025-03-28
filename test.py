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
from utils import RoadSegmentationModel
    
    
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

img_path = "test_img2.jpg"
img = Preprocess(img_path)
device = torch.device(("mps"))
model.to(device)
img = img.to(device)

with torch.no_grad():
    out = model(img)
yhat = out.squeeze().cpu().numpy()

def overlay_mask(image_path, mask, alpha=0.5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))


    color_mask = np.zeros_like(image)
    color_mask[:, :, 2] = mask * 255
    blended = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    return blended


segmentation_overlay = overlay_mask("test_img2.jpg", yhat)
y = cv2.imread(img_path,cv2.COLOR_BGR2RGB)
y = cv2.resize(img,(256,256))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(y, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(segmentation_overlay, cmap="gray")