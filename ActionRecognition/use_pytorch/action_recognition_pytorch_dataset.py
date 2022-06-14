import os
import random

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets.folder import make_dataset
from torchvision import transforms as t
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

        voutputFrame = frames[startFrameNumber].unsqueeze(dim = 0)
        
        for frame in frames[startFrameNumber + 1 : startFrameNumber + self.clip_len]:
            voutputFrame = torch.vstack((voutputFrame, frame.unsqueeze(dim = 0)))
        
        return voutputFrame, target





transforms = [t.Resize((250, 250))]
frame_transform = t.Compose(transforms)

dataset = VideoDataset("./dataset", frame_transform=frame_transform,)

loader = DataLoader(dataset, batch_size=3, shuffle=True)

data = {"video": [], 'start': [], 'end': [], 'tensorsize': []}

for batch in tqdm(loader):
    
    print(batch.shape())
    #for i in range(len(batch['path'])):
     #   data['video'].append(batch['path'][i])
    #    data['start'].append(batch['start'][i].item())
    #    data['end'].append(batch['end'][i].item())
    #    data['tensorsize'].append(batch['video'][i].size())
        

#print(data)