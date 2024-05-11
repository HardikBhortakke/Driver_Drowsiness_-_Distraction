import cv2
import numpy as np
import socket
import struct

# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
server_address = ('127.0.0.1', 6000)
client_socket.connect(server_address)

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
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Display the frame
        cv2.imshow('Display Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print("Error receiving frame:", e)
        break

# Close the connection
client_socket.close()
cv2.destroyAllWindows()
