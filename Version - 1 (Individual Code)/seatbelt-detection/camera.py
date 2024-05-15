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


print("Script loaded. Import complete")

OBJECT_DETECTION_MODEL_PATH = "models/best.pt"
PREDICTOR_MODEL_PATH = "models/keras_model.h5"
CLASS_NAMES = {0: 'No Seatbelt worn', 1: 'Seatbelt Worn'}

THRESHOLD_SCORE = 0.99

SKIP_FRAMES = 20  # skips every 2 frames
MAX_FRAME_RECORD = 500
# OUTPUT_FILE = './Results/test_result_' + dt.datetime.strftime(dt.datetime.now(), "%Y%m%d%H%M%S") + '.mp4'

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)

def send_message(message):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 9999))
    client_socket.send(message.encode())
    client_socket.close()

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

# cap = cv2.VideoCapture(0)  # 0 for default webcam
cap = cv2.VideoCapture(0, cv2.CAP_LIBCAMERA)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_MODE, 1)
cap.set(cv2.CAP_PROP_FORMAT, 0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
fps = 30  # Assuming 30 fps for webcam input

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, size)

print("Analyzing webcam input...")

frame_count = 0
img_resize = np.zeros((int(frame_height/2), int(frame_width/2), 3), dtype=np.uint8)
img_output = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
start_time = time.time()

print("Started")
while True:
    ret, img = cap.read()

    if ret:
        frame_count += 1

        if frame_count % SKIP_FRAMES == 0:
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
                    send_message("c = True")
                elif y_pred == CLASS_NAMES[1]:
                    draw_color = COLOR_GREEN
                    print("Seatbelt Worn")
                    send_message("c = False")

                if score >= THRESHOLD_SCORE:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f'{y_pred} {str(score)[:4]}', (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                draw_color, 2)

            img_output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_resize = cv2.resize(img_output, (int(frame_width/2), int(frame_height/2)))
            #cv2.imshow('Video', img_resize)
            
        cv2.imshow('Video', img_resize)
        # out.write(img_output)


    # else:
    #     break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
print("Total Time = ", (end_time - start_time))

cap.release()
# out.release()
cv2.destroyAllWindows()

# print("Script run complete. Results saved to :", OUTPUT_FILE)
