import torch.onnx 
import torch
import torchvision
import cv2
import argparse
import time
import numpy as np
import torch.nn as nn
import pickle
import os
from torchvision import transforms as t

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', dest = 'input_folder', help='input_folder', type=str, default='C:\\Users\\Donghee Han\\Desktop\\CMU_program\\IITP 팀플\\workspace\\life-logging-system\\ActionRecognition\\use_pytorch\\demoinput')
parser.add_argument('-c', '--clip-len', dest='clip_len', default=16, type=int,
                    help='number of frames to consider for each prediction')
parser.add_argument('-m', '--model-path', dest='mode_path', default='C:\\Users\\Donghee Han\\Desktop\\CMU_program\\IITP 팀플\\workspace\\mc3_18_0.77.pkl', type=str,
                    help='input pretrain model pth path')

parser.add_argument('-o', '--onnx-path', dest='onnx_model_path', default="actionrecognition_mc18.onnx", type=str,
                    help='input onnx model path')


args = vars(parser.parse_args())
#### PRINT INFO #####
print(f"Number of frames to consider for each prediction: {args['clip_len']}")

MODELPATH = args['mode_path']
INPUTSPATH = args['input_folder']
ONNXPATH = args['onnx_model_path']

# get the lables
labels = ['fight', 'nofight']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
# load the model
model = torchvision.models.video.mc3_18(pretrained=True, progress=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(labels))

model.load_state_dict(torch.load(MODELPATH))
# load the model onto the computation device
model = model.eval()
model = model.to(device)
converprob = nn.Softmax()

# if use webcam, 
#useWabCam = True

# if load some videos
useWabCam = False

output_fps = 10

count = 0

frame_action_pair = []

transforms = [t.Resize((112, 112))]
frame_transform = t.Compose(transforms)

if useWabCam == False:
    
    videofileList = os.listdir(INPUTSPATH)
    for vieofile in videofileList:
        videoPath = INPUTSPATH + '/' + vieofile

        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')

        # get the frame width and height
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        save_name = f"{videoPath.split('/')[-1].split('.')[0]}"

        # define codec and create VideoWriter object 
        out = cv2.VideoWriter(f"./demooutput/{save_name}.mp4", 
                            cv2.VideoWriter_fourcc(*'mp4v'), output_fps, 
                            (frame_width, frame_height))

        frame_count = 0 # to count total frames
        total_fps = 0 # to get the final frames per second

        # a clips list to append and store the individual frames
        clips = []              

        # read until end of video
        while(cap.isOpened()):
            # capture each frame of the video
            ret, frame = cap.read()
            if ret == True:
                
                image = frame.copy()
                
                frame = frame_transform(torch.Tensor(frame).permute(2,0,1))
                
                clips.append(frame)

                if len(clips) == args['clip_len']:
                    with torch.no_grad(): # we do not want to backprop any gradients
                        
                        frames = clips
                        voutputFrame = frames[0].unsqueeze(dim = 0)

                        for index in range(0 + 1, len(clips)):
                            voutputFrame = torch.vstack((voutputFrame, frames[index].unsqueeze(dim = 0)))
                        
                        voutputFrame = voutputFrame.permute(1,0,2,3).to(device)
                        voutputFrame = voutputFrame.unsqueeze(dim = 0)
                        # forward pass to get the predictions

                        inputData = voutputFrame
                        outputs = model(voutputFrame)

                        # Export the model   
                        torch.onnx.export(model,         # model being run 
                            inputData,       # model input (or a tuple for multiple inputs) 
                            ONNXPATH,       # where to save the model  
                            export_params=True,  # store the trained parameter weights inside the model file 
                            opset_version=10,    # the ONNX version to export the model to 
                            do_constant_folding=True,  # whether to execute constant folding for optimization 
                            input_names = ['modelInput'],   # the model's input names 
                            output_names = ['modelOutput'], # the model's output names 
                            dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                                    'modelOutput' : {0 : 'batch_size'}}) 
                        print(" ") 
                        print('Model has been converted to ONNX') 

                        break
        break            

                                  
    cap.release()



    