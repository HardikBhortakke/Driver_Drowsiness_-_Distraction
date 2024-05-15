import cv2
import numpy as np
import socket
import struct 

cap=cv2.VideoCapture(2)

#cap = cv2.VideoCapture(0, cv2.CAP_LIBCAMERA)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
#cap.set(cv2.CAP_PROP_MODE, 1)
#cap.set(cv2.CAP_PROP_FORMAT, 0)

clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('127.0.0.1',5000))

while True:
    ret,frame=cap.read()
    if ret:
        data = cv2.imencode('.jpg', frame)[1].tobytes()
        clientsocket.sendall(struct.pack("L", len(data))+data)
        print(f"Sent frame: {len(data)} bytes")
        #cv2.imshow('Webcam Frame', frame)
        cv2.waitKey(1)
