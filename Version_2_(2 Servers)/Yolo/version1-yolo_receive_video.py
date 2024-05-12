from ultralytics import YOLO
import cv2
import math
import socket
import time
import numpy as np
import struct

# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
server_address = ('127.0.0.1', 6000)
client_socket.connect(server_address)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket1.connect(("localhost", 7000))

def send_message(message):

    client_socket1.send(message.encode())


n1 = 0

while True:
    try:
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

        results = model(img, stream=True)
        a = 0
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                if (classNames[cls] == "cell phone"):
                    a = 1

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        if (a == 1):
            print("Cellphone Detected")
            n1 = 100
        else:
            if (n1 > 0):
                n1 -= 1

        if (n1 > 0):
            send_message("d = True")
        else:
            send_message("d = False")

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as e:
        print("Error receiving frame:", e)
        break

# Close the connection
client_socket.close()
client_socket1.close()

cv2.destroyAllWindows()
