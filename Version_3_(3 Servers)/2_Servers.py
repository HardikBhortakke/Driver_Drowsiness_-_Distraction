import socket
import struct
import cv2
import numpy as np
import threading
from queue import Queue

def free_port(port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", port))
        print(f"Port {port} is now free.")
    except OSError as e:
        print(f"Failed to free port {port}: {e}")
    finally:
        sock.close()

free_port(5000)
free_port(6000)

# Initialize frame queue
frame_queue = Queue(maxsize=50)  # Adjust maxsize as needed

# Server 1 settings
SERVER1_HOST = '127.0.0.1'
SERVER1_PORT = 5000

# Server 2 settings
SERVER2_HOST = '127.0.0.1'
SERVER2_PORT = 6000

# Function to continuously receive frames from clients and store them in the queue
def manage_client_server1(conn):
    while True:
        data = b''
        payload_size = struct.calcsize("L") 
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        frame_queue.put(frame_data)

# Server 1 thread function
def server1_thread():
    try:
        server1_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server1_socket.bind((SERVER1_HOST, SERVER1_PORT))
        server1_socket.listen(1)
        conn, addr = server1_socket.accept()
        print(f"Connected to {addr}")
        manage_client_server1(conn)
    except KeyboardInterrupt:
        print("Server 1 is shutting down...")
    finally:
        server1_socket.close()

# Function to continuously send frames to connected clients
def send_frames_to_clients():
    while True:
        frame_data = frame_queue.get()
        for client_socket in connected_clients:
            try:
                client_socket.sendall(struct.pack("L", len(frame_data)) + frame_data)
                print(f"Sent frame: {len(frame_data)} bytes")
            except Exception as e:
                print(f"Error sending frame to client: {e}")

# Server 2 thread function
def server2_thread():
    try:
        global connected_clients
        connected_clients = []
        server2_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server2_socket.bind((SERVER2_HOST, SERVER2_PORT))
        server2_socket.listen(3)  # Listen for up to 3 clients
        send_frames_thread = threading.Thread(target=send_frames_to_clients)
        send_frames_thread.start()
        while True:
            conn, addr = server2_socket.accept()
            print(f"Connected to {addr}")
            connected_clients.append(conn)
    except KeyboardInterrupt:
        print("Server 2 is shutting down...")
        # Close the server socket
        server2_socket.close()
    finally:
        # Close any remaining client connections
        for client_socket in connected_clients:
            client_socket.close()


# Start server threads
server1_thread = threading.Thread(target=server1_thread)
server1_thread.start()

server2_thread = threading.Thread(target=server2_thread)
server2_thread.start()

# Wait for server threads to finish
#server1_thread.join()
#server2_thread.join()

print("All servers are now closed.")
