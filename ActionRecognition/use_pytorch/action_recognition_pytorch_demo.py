import torch
import torchvision
import cv2
import argparse
import time
import numpy as np
import utils
import torch.nn as nn
import pickle
import os

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', dest='input', help='path to input video', type=str, default='./demopretraininput')
parser.add_argument('-c', '--clip-len', dest='clip_len', default=16, type=int,
                    help='number of frames to consider for each prediction')

args = vars(parser.parse_args())
#### PRINT INFO #####
print(f"Number of frames to consider for each prediction: {args['clip_len']}")

INPUTSPATH = args['input']

# get the lables
class_names = utils.class_names
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
model = torchvision.models.video.mc3_18(pretrained=True, progress=True)

# load the model onto the computation device
model = model.eval().to(device)
print('model mode change (eval!!)')

# convert to Probability from model ouputs
converprob = nn.Softmax()

output_fps = 10

count = 0

videofileList = os.listdir(INPUTSPATH)
for vieofile in videofileList:
    frame_action_pair = []
    videoPath = INPUTSPATH + '/' + vieofile
    cap = cv2.VideoCapture(videoPath)

    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    save_name = f"{videoPath.split('/')[-1].split('.')[0]}"
    # define codec and create VideoWriter object 
    out = cv2.VideoWriter(f"demopretrainoutput/{save_name}.mp4", 
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
            # get the start time
            start_time = time.time()
            image = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = utils.transform(frame)
            frame = np.array(frame)
            clips.append(frame)
            if len(clips) == args['clip_len']:
                with torch.no_grad(): # we do not want to backprop any gradients
                    input_frames = np.array(clips)
                    # add an extra dimension        
                    input_frames = np.expand_dims(input_frames, axis=0)

                    # transpose to get [1, 3, num_clips, height, width]
                    input_frames = np.transpose(input_frames, (0, 2, 1, 3, 4))

                    # convert the frames to tensor
                    input_frames = torch.tensor(input_frames, dtype=torch.float32)
                    input_frames = input_frames.to(device)
                    # forward pass to get the predictions
                    outputs = model(input_frames)
                        
                    outputs = converprob(outputs)

                    # get the prediction index
                    candidates_Value_Index_Pair = [] 
                    candidatesLabel = [] 

                    for _ in range(1):
                        val, preds = torch.max(outputs, 1)
                        outputs[0][preds] = -10000.0
                        candidates_Value_Index_Pair.append((val,preds))

                    for value, index in candidates_Value_Index_Pair:
                        # map predictions to the respective class names
                        candidatesLabel.append((value,class_names[index].strip()))
        

                # get the end time
                end_time = time.time()

                # get the fps
                fps = 1 / (end_time - start_time)
                    
                # add fps to total fps
                total_fps += fps

                # increment frame count
                frame_count += 1
                wait_time = max(1, int(fps/4))

                # draw fps, label
                cv2.putText(image, 'fps : ' +str(int(fps)), (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (22, 22, 222), 2, 
                            lineType=cv2.LINE_AA)

                currentVideoTime = frame_count / output_fps

                cv2.putText(image, 'time : ' +str(round(currentVideoTime,2)), (130, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (22, 22, 222), 2, 
                            lineType=cv2.LINE_AA)

                frame_action_pair.append((frame_count, candidatesLabel[0][1]))

                for index, (value, label) in enumerate(candidatesLabel):
                    color = (22,22,222)
    
                    cv2.putText(image, 'action : ' + '[' + str(round(value.item(),3)) + '] ' + label, (15, (index * 35) + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                                    lineType=cv2.LINE_AA)


                clips.pop(0)
                cv2.imshow('image', image)
                out.write(image)
                    
                # press `q` to exit
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
        else:
            break

    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

    # save
    with open('meta_data.pickle', 'wb') as f:
        pickle.dump(frame_action_pair, f, pickle.HIGHEST_PROTOCOL)

    # load
    with open('meta_data.pickle', 'rb') as f:
        data = pickle.load(f)

    print(data)
