import os
import random

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets.folder import make_dataset
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim
import cv2

import itertools
from tqdm import tqdm
import numpy as np


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_samples(root, extensions=(".mp4", ".avi")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, frame_transform=None, clip_len=16):
        self.samples = get_samples(root)
        self.clip_len = clip_len
        self.frame_transform = frame_transform
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, i):
        
        # Get random sample
        path, target = self.samples[i]
        cap = cv2.VideoCapture(path)

        frames = []

        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                frames.append(torch.Tensor(frame).permute(2,0,1))
            else:
                break
        
        startFrameNumber = random.randint(0, len(frames) - self.clip_len)


        #image preprocessing
        if  self.frame_transform != None:
            for index in range(startFrameNumber, startFrameNumber + self.clip_len):
                frames[index] = self.frame_transform(frames[index])

        #adapt vstack
        voutputFrame = frames[startFrameNumber].unsqueeze(dim = 0)
        for index in range(startFrameNumber + 1, startFrameNumber + self.clip_len):
            voutputFrame = torch.vstack((voutputFrame, frames[index].unsqueeze(dim = 0)))
        
        voutputFrame = voutputFrame.permute(1,0,2,3)
        return voutputFrame, target


transforms = [t.Resize((128, 171))]
frame_transform = t.Compose(transforms)

dataset = VideoDataset("./dataset", frame_transform=frame_transform,)

loader = DataLoader(dataset, batch_size=10, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 0 Fi
labels = [0, 1]
# load pretrained model
model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(labels))

# load the model onto the computation device
model = model.to(device)



epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01)

for epoch in range(epochs):
    running_loss = 0
    running_corrects = 0

    batch_bar = tqdm(total=len(loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 
    for batch in loader:

        voutputFrame, target = batch
        voutputFrame = voutputFrame.to(device)
        target = target.to(device)

                # zero the parameter gradients
        optimizer.zero_grad()
        # forward

        prediction = model(voutputFrame)
        _, preds = torch.max(prediction, 1)
        loss = criterion(prediction, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(preds == target.data)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix( 
            num_correct = f"{running_corrects}", 
            loss=f"{loss.item():.4f}", 
            lr=f"{float(optimizer.param_groups[0]['lr']):.4f}")

        batch_bar.update() # Update tqdm bar
    batch_bar.close() # You need this to close the tqdm bar

    epoch_loss = running_loss / len(loader)
    epoch_acc = running_corrects.double() / len(loader)

    print(f'{epoch} epoch summary Loss: {epoch_loss:.2f} Acc: {epoch_acc:.2f}')
 