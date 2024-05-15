# How to run code python3 video_drowsiness_detection_withclient.py --shape_predictor shape_predictor_68_face_landmarks.dat
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

def send_message(message):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 9999))
    client_socket.send(message.encode())
    client_socket.close()

# style.use('fivethirtyeight')
# # Creating the dataset 
# def assure_path_exists(path):
#     dir = os.path.dirname(path)
#     if not os.path.exists(dir):
#         os.makedirs(dir)

cap = cv2.VideoCapture(0, cv2.CAP_LIBCAMERA)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_MODE, 1)
cap.set(cv2.CAP_PROP_FORMAT, 0)


#all eye  and mouth aspect ratio with time
# ear_list=[]
# total_ear=[]
# mar_list=[]
# total_mar=[]
# ts=[]
# total_ts=[]
# Construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser() 
ap.add_argument("-p", "--shape_predictor", required = True, help = "path to dlib's facial landmark predictor")
ap.add_argument("-r", "--picamera", type = int, default = -1, help = "whether raspberry pi camera shall be used or not")
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

# Now start the video stream and allow the camera to warm-up
# Open input video file
# input_video = "./Videos/test2.mp4"
# cap = cv2.VideoCapture(input_video)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

#output_video = "./output_video.mp4"
# Modify this line to try different FourCC codes
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Check if the video file is opened successfully
# if not cap.isOpened():
#     print("Error: Unable to open input video file.")
#     exit()

count_sleep = 0
count_yawn = 0
flag = 10
# Now, loop over all the frames and detect the faces
while True: 
	# Extract a frame 
    success, frame = cap.read()
    a = 0
    # Check if frame is read successfully
    if success:
        
        cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3) 
        # Resize the frame 
        frame = imutils.resize(frame, width = 500)
        # Convert the frame to grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces 
        rects = detector(frame, 1)
        RECTS = list(rects)

        if(len(RECTS) == 0):
            print("DISTRACTED!")
            # cv2.putText(frame, "DISTRACTED", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            send_message("a = True")
        # Now loop over all the face detections and apply the predictor 
        for (i, rect) in enumerate(rects): 
            shape = predictor(gray, rect)
            # Convert it to a (68, 2) size numpy array 
            shape = face_utils.shape_to_np(shape)

            # Draw a rectangle over the detected face 
            # (x, y, w, h) = face_utils.rect_to_bb(rect) 
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)	
            # Put a number 
            # cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            leftEye = shape[lstart:lend]
            rightEye = shape[rstart:rend] 
            mouth = shape[mstart:mend]
            # Compute the EAR for both the eyes 
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Take the average of both the EAR
            EAR = (leftEAR + rightEAR) / 2.0
            #live datawrite in csv
            # ear_list.append(EAR)
            #print(ear_list)
            

            # ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
            # Compute the convex hull for both the eyes and then visualize it
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # Draw the contours 
            # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

            MAR = mouth_aspect_ratio(mouth)
            # mar_list.append(MAR/10)
            # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
            # Thus, count the number of frames for which the eye remains closed 
            if EAR < EAR_THRESHOLD: 
                FRAME_COUNT += 1

                # cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                # cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

                if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
                    count_sleep += 1
                    # Add the frame to the dataset ar a proof of drowsy driving
                    #cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
                    #playsound('sound files/alarm.mp3')
                    # cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # cv2.putText(frame, "DON'T SLEEP!", (270, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("DON'T SLEEP!")
                    a = 1
                    send_message("a = True")
            else: 	
                FRAME_COUNT = 0
                send_message("a = False")
            #cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check if the person is yawning
            if MAR > MAR_THRESHOLD:
                count_yawn += 1
                # cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
                # cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # print("DROWSINESS ALERT!")
                # Add the frame to the dataset ar a proof of drowsy driving
                #cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
                #playsound('sound files/alarm.mp3')
                send_message("b = True")
                if (a == 0):
                    print("DON'T YAWN!")
                    # cv2.putText(frame, "DON'T YAWN!", (270, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                send_message("b = False")
        for i in ear_list:
            total_ear.append(i)
        for i in mar_list:
            total_mar.append(i)			
        for i in ts:
            total_ts.append(i)
        #display the frame 
        img_resize = cv2.resize(frame, (int(frame_width/2), int(frame_height/2)))
        cv2.imshow("Output", img_resize)
        #out.write(frame)

        key = cv2.waitKey(1) & 0xFF 

        if key == ord('q'):
            break

# a = total_ear
# b=total_mar
# c = total_ts

# df = pd.DataFrame({"EAR" : a, "MAR":b,"TIME" : c})
# df.to_csv("op_webcam.csv", index=False)
# df=pd.read_csv("op_webcam.csv")

# df.plot(x='TIME',y=['EAR','MAR'])
# #plt.xticks(rotation=45, ha='right')

# plt.subplots_adjust(bottom=0.30)
# plt.title('EAR & MAR calculation over time of webcam')
# plt.ylabel('EAR & MAR')
# plt.gca().axes.get_xaxis().set_visible(False)
# plt.show()

# out.release()

cv2.destroyAllWindows()
cap.release()
