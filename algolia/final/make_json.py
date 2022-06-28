import json
import os

json_path = 'temp.json'
filename, fps, frame, timestamp, face, face_distance, face_attributes, action, action_score = None, None, None, None, None, None, None, None, None
metadata = []



########################## 여기에 코드를 넣어주세요! ##########################

filename  = '100.wav'
fps, frame, timestamp, face, face_distance= 1, 1, 1, 1, 1
face_attributes = ['Beard', 'Yellow hair']
# store metadata in dictionary
data = {}
data["filename"] = filename
data["fps"] = fps
data["frame"] = frame
data["timestamp(s)"] = timestamp
data["face"] = face
data["face_distance"] = face_distance
data["face_attributes"] = face_attributes
data["action"] = action
data["action_score"] = action_score
metadata.append(data)

filename  = '101.wav'
fps, frame, timestamp, face, face_distance= 1, 1, 1, 1, 1
face_attributes = ['Brown Eyes', 'Yellow hair']
# store metadata in dictionary
data = {}
data["filename"] = filename
data["fps"] = fps
data["frame"] = frame
data["timestamp(s)"] = timestamp
data["face"] = face
data["face_distance"] = face_distance
data["face_attributes"] = face_attributes
data["action"] = action
data["action_score"] = action_score
metadata.append(data)

# write new data to json file
with open(json_path, 'w') as write_file:
    json.dump(metadata, write_file)
write_file.close()
