import cv2
import os
import datetime as dt
import numpy as np
import tensorflow as tf
import torch
from keras.models import load_model
from PIL import Image
import socket
import time
import struct

# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
server_address = ('127.0.0.1', 6000)
client_socket.connect(server_address)

print("Script loaded. Import complete")

OBJECT_DETECTION_MODEL_PATH = "models/best.pt"
PREDICTOR_MODEL_PATH = "models/keras_model.h5"
CLASS_NAMES = {0: 'No Seatbelt worn', 1: 'Seatbelt Worn'}

THRESHOLD_SCORE = 0.99

SKIP_FRAMES = 20  # skips every 2 frames
MAX_FRAME_RECORD = 500

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)

def send_message(message):
    client_socket.send(message.encode())

def prediction_func(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = (img / 127.5) - 1
    img = tf.expand_dims(img, axis=0)
    pred = predictor.predict(img)
    index = np.argmax(pred)
    class_name = CLASS_NAMES[index]
    confidence_score = pred[0][index]
    return class_name, confidence_score

predictor = load_model(PREDICTOR_MODEL_PATH, compile=False)
print("Predictor loaded")

# Ultralytics object detection model : https://docs.ultralytics.com/yolov5/
model = torch.hub.load("ultralytics/yolov5", "custom", path=OBJECT_DETECTION_MODEL_PATH, force_reload=False)

print("Analyzing webcam input...")

frame_count = -1
img_resize = np.zeros((100, 100, 3), dtype=np.uint8)
img_output = np.zeros((100, 100, 3), dtype=np.uint8)

print("Started")
while True:
    try:
        frame_count += 1
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
        img = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img: 
            if frame_count % SKIP_FRAMES == 0:

                frame_height, frame_width = img.shape[:2]

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = model(img)

                boxes = results.xyxy[0]
                boxes = boxes.cpu()
                for j in boxes:
                    x1, y1, x2, y2, score, y_pred = j.numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    img_crop = img[y1:y2, x1:x2]

                    y_pred, score = prediction_func(img_crop)

                    if y_pred == CLASS_NAMES[0]:
                        draw_color = COLOR_RED
                        print("XXXXX Wear Seatbelt XXXXX")
                        #send_message("c = True")
                    elif y_pred == CLASS_NAMES[1]:
                        draw_color = COLOR_GREEN
                        print("Seatbelt Worn")
                        #send_message("c = False")

                    if score >= THRESHOLD_SCORE:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, f'{y_pred} {str(score)[:4]}', (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    draw_color, 2)

                img_output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_resize = cv2.resize(img_output, (int(frame_width/2), int(frame_height/2)))
                
        cv2.imshow('Video', img_resize)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error receiving frame:", e)
        break

client_socket.close()
cv2.destroyAllWindows()

