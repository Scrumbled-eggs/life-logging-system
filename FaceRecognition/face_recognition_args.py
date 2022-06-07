import face_recognition
import cv2
import numpy as np
import argparse
import pickle

def main(args):
    video = args.input
    video_capture = cv2.VideoCapture(video)

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    out = cv2.VideoWriter(args.output, fourcc, args.fps, (frame_width, frame_height))
    #fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Load a sample picture and learn how to recognize it.
    dain_image = face_recognition.load_image_file("pics/dain.jpeg")
    dain_face_encoding = face_recognition.face_encodings(dain_image)[0]

    chanyang_image = face_recognition.load_image_file("pics/chanyang.jpg")
    chanyang_face_encoding = face_recognition.face_encodings(chanyang_image)[0]

    tom_image = face_recognition.load_image_file("pics/tom.jpg")
    tom_face_encoding = face_recognition.face_encodings(tom_image)[0]

    # Load a second sample picture and learn how to recognize it.
    anthony_image = face_recognition.load_image_file("pics/anthony.jpg")
    anthony_face_encoding = face_recognition.face_encodings(anthony_image)[0]

    # Load a third sample picture and learn how to recognize it.
    val_image = face_recognition.load_image_file("pics/val.jpg")
    val_face_encoding = face_recognition.face_encodings(val_image)[0]

    # Load a fourth sample picture and learn how to recognize it.
    skerritt_image = face_recognition.load_image_file("pics/skerritt.jpg")
    skerritt_face_encoding = face_recognition.face_encodings(skerritt_image)[0]
    # Create arrays of known face encodings and their names
    known_face_encodings = [
        chanyang_face_encoding,
        dain_face_encoding,
        tom_face_encoding,
        anthony_face_encoding,
        val_face_encoding,
        skerritt_face_encoding
    ]
    known_face_names = [
        "Chanyang Seo",
        "Dain Kim",
        "Tom Cruise",
        "Anthony Edwards",
        "Val Kilmer",
        "Tom Skerritt"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    cnt = 0
    time = 0
    frame_info = dict()
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if not ret:
            break
        cnt += 1
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame
        
        time = cnt/args.fps
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, 'frame no.:%d time:%0.4f sec'%(cnt, time), (50, 50), font, 1.0, (255, 255, 255), 1)
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, 'name:%s'%(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        #cv2.imshow('Video', frame)
        out.write(frame)
        frame_info[cnt] = face_names
        
    # Release handle to the webcam
    video_capture.release()
    
    fname = args.input.split('.')[-2]
    with open('%s_frame_info.pkl'%fname, 'wb') as f:
        pickle.dump(frame_info, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, help="path to output video file")
    parser.add_argument("-i", "--input", required=True, help="path to input video file")
    parser.add_argument("-f", "--fps", default=15.0, type=float, help='target fps for the output video')
    args = parser.parse_args()

    main(args)