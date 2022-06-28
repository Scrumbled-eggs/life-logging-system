from pathlib import Path
import json

json_path = 'temp.json'
filename, fps, frame, timestamp, face, face_distance, Beard, action, action_score = None, None, None, None, None, None, None, None, None
metadata = {}


### 여기에 코드를 넣어주세요!





# store metadata in dictionary
metadata["filename"] = filename
metadata["fps"] = fps
metadata["frame"] = frame
metadata["timestamp(s)"] = timestamp
metadata["face"] = face
metadata["face_distance"] = face_distance
metadata["Beard"] = Beard
metadata["action"] = action
metadata["action_score"] = action_score


# read existing json file
if os.path.exists(json_path):
    with open(json_path, 'r') as read_file:
        json_data = json.load(read_file)
    read_file.close()
else:
    # create a new json file
    Path.mkdir(Path(movieJsonDir), exist_ok=True)

# write new data to json file
with open(json_path, 'w') as write_file:
    json.dumps(metadata)
write_file.close()
