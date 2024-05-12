# How to run code python3 websocket_feed.py --shape_predictor shape_predictor_68_face_landmarks.dat
# Import the necessary packages 
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from EAR_calculator import *
from imutils import face_utils 
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from matplotlib import style 
import imutils 
import dlib
import time 
import argparse 
import cv2 
from scipy.spatial import distance as dist
import os 
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import socket
import struct
import threading


style.use('fivethirtyeight')
# Creating the dataset 
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


#all eye  and mouth aspect ratio with time
ear_list=[]
total_ear=[]
mar_list=[]
total_mar=[]
ts=[]
total_ts=[]
# Construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser() 
ap.add_argument("-p", "--shape_predictor", required = True, help = "path to dlib's facial landmark predictor")
args = vars(ap.parse_args())

# Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink 
EAR_THRESHOLD = 0.3
# Declare another costant to hold the consecutive number of frames to consider for a blink 
CONSECUTIVE_FRAMES = 20 
# Another constant which will work as a threshold for MAR value
MAR_THRESHOLD = 14

# Initialize two counters 
BLINK_COUNT = 0 
FRAME_COUNT = 0 

# Now, intialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
print("[INFO]Loading the predictor.....")
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(args["shape_predictor"])

# Grab the indexes of the facial landamarks for the left and right eye respectively 
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

count_sleep = 0
count_yawn = 0
flag = 10
n1 = 0
count = 0
skip_frames = 5

# Function to send a message to the client
def send_message(message, client_socket):
    try:
        client_socket.send(message.encode())
    except Exception as e:
        print("Error sending message:", e)

# Function to process frames from a client
def process_frame(client_socket, n):
    global BLINK_COUNT, FRAME_COUNT, count_sleep, count_yawn, count, skip_frames
    while True:
        try:
            a = 0

            # Receive frame size
            size_data = client_socket.recv(struct.calcsize("L"))
            if not size_data:
                break
            size = struct.unpack("L", size_data)[0]

            # Receive frame data
            frame_data = b''
            while len(frame_data) < size:
                packet = client_socket.recv(size - len(frame_data))
                if not packet:
                    break
                frame_data += packet

            # Convert frame data to numpy array and decode it
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3) 
            # Resize the frame 
            frame = imutils.resize(frame, width = 500)
            # Convert the frame to grayscale 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #if (count == 0):
                # Detect faces 
            rects = detector(frame, 1)
            #print(rects)
            RECTS = list(rects)
            #print(RECTS)
            #count = skip_frames
            #else:
            #count -= 1

            if(len(RECTS) == 0):
                cv2.putText(frame, "DISTRACTED", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                send_message("a = True", client_socket2)
            # Now loop over all the face detections and apply the predictor 
            for (i, rect) in enumerate(rects): 
                shape = predictor(gray, rect)
                # Convert it to a (68, 2) size numpy array 
                shape = face_utils.shape_to_np(shape)

                # Draw a rectangle over the detected face 
                (x, y, w, h) = face_utils.rect_to_bb(rect) 
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)	
                # Put a number 
                cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                leftEye = shape[lstart:lend]
                rightEye = shape[rstart:rend] 
                mouth = shape[mstart:mend]
                # Compute the EAR for both the eyes 
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # Take the average of both the EAR
                EAR = (leftEAR + rightEAR) / 2.0
                
                # Compute the convex hull for both the eyes and then visualize it
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                # Draw the contours 
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

                MAR = mouth_aspect_ratio(mouth)
                # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
                # Thus, count the number of frames for which the eye remains closed 
                if EAR < EAR_THRESHOLD: 
                    FRAME_COUNT += 1

                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

                    if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
                        count_sleep += 1
                        cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "DON'T SLEEP!", (270, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        a = 1
                        send_message("a = True", client_socket2)
                else: 	
                    FRAME_COUNT = 0
                    send_message("a = False", client_socket2)
                #cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Check if the person is yawning
                if MAR > MAR_THRESHOLD:
                    count_yawn += 1
                    cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
                    cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    send_message("b = True", client_socket2)
                    if (a == 0):
                        cv2.putText(frame, "DON'T YAWN!", (270, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    send_message("b = False", client_socket2)
            #display the frame 
            cv2.imshow("Output", frame)
            key = cv2.waitKey(1) & 0xFF 
        
        

            if key == ord('q'):
                break

        except Exception as e:
            print("Error receiving frame:", e)
            break

# Create sockets for the clients
client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address1 = ('127.0.0.1', 6000)
client_socket1.connect(server_address1)

client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address2 = ('localhost', 7000)
client_socket2.connect(server_address2)
    

# Create threads for processing frames from each client
thread1 = threading.Thread(target=process_frame, args=(client_socket1, n1))
thread2 = threading.Thread(target=process_frame, args=(client_socket2, n1))

# Start the threads
thread1.start()
thread2.start()

# Wait for threads to finish
thread1.join()
thread2.join()

# Close the connections
client_socket1.close()
client_socket2.close()

cv2.destroyAllWindows()
