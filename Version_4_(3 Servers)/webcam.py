import cv2
import numpy as np
import socket
import struct 

cap=cv2.VideoCapture(0)
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('127.0.0.1',5000))

while True:
    ret,frame=cap.read()
    data = cv2.imencode('.jpg', frame)[1].tobytes()
    clientsocket.sendall(struct.pack("L", len(data))+data)
    print(f"Sent frame: {len(data)} bytes")
    cv2.imshow('Webcam Frame', frame)
    cv2.waitKey(1)
