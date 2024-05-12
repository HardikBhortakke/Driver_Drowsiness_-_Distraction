from ultralytics import YOLO
import cv2
import math
import socket
import time
import numpy as np
import struct
import threading

# Create a socket for server 1
client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address1 = ('127.0.0.1', 6000)
client_socket1.connect(server_address1)

# Create a socket for server 2
client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address2 = ('localhost', 7000)
client_socket2.connect(server_address2)

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
              "teddy bear", "hair drier", "toothbrush"]

n1 = 0  # Define n1 here

def send_message(message, client_socket):
    try:
        client_socket.send(message.encode())
    except Exception as e:
        print("Error sending message:", e)


def process_frame(client_socket, n):
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
                n = 100
            else:
                if (n > 0):
                    n -= 1

            if (n > 0):
                send_message("d = True", client_socket2)
            else:
                send_message("d = False", client_socket2)

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == ord('q'):
                break

        except Exception as e:
            print("Error receiving frame:", e)
            break


# Create threads for processing frames from each server
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
