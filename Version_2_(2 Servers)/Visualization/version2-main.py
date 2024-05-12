import sys
import socket
import threading
import struct
from queue import Queue
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from ess import Ui_MainWindow  # Importing the generated UI class

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
free_port(7000)


class Communicate(QObject):
    update_signal = pyqtSignal(bool, bool, bool, bool)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Initialize variables
        self.a = False
        self.b = False
        self.c = False
        self.d = False

        # Create communication object
        self.communicate = Communicate()
        self.communicate.update_signal.connect(self.update_variables)

        # Start timer to update variables and check conditions
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.send_signal)
        self.timer.start(100)  # Update every 10 milliseconds

        # Initialize frame queue
        self.frame_queue = Queue(maxsize=5)  # Adjust maxsize as needed
        self.connected_clients = []

        # Server 1 settings
        self.SERVER1_HOST = '127.0.0.1'
        self.SERVER1_PORT = 5000

        # Server 2 settings
        self.SERVER2_HOST = '127.0.0.1'
        self.SERVER2_PORT = 6000

        # Start server to listen to clients
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.start()

        self.server1_thread = threading.Thread(target=self.server1_thread)
        self.server1_thread.start()

        self.server2_thread = threading.Thread(target=self.server2_thread)
        self.server2_thread.start()

    def send_signal(self):
        self.communicate.update_signal.emit(self.a, self.b, self.c, self.d)

    def update_variables(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        # Check conditions for showing icons and warning text
        if self.a:
            self.ui.drowsy_icon.setPixmap(QPixmap("drowsy.png"))
        else:
            self.ui.drowsy_icon.clear()

        if self.b:
            self.ui.yawn_icon.setPixmap(QPixmap("yawning.png"))
        else:
            self.ui.yawn_icon.clear()

        if self.c:
            self.ui.seatbelt_icon.setPixmap(QPixmap("seatbelt.png"))
        else:
            self.ui.seatbelt_icon.clear()

        if self.d:
            self.ui.mobile_distraction_icon.setPixmap(QPixmap("mobile_distraction.png"))
        else:
            self.ui.mobile_distraction_icon.clear()

        # Check if any icon is shown, then show warning text
        if self.a or self.b or self.c or self.d:
            self.ui.display_text.setText("<html><head/><body><p align=\"center\"><span style=\" font-size:36pt; color:#e32f17;\">WARNING!</span></p></body></html>")
        else:
            self.ui.display_text.clear()

    # Function to continuously receive frames from clients and store them in the queue
    def manage_client_server1(self, conn):
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
            self.frame_queue.put(frame_data)

    # Server 1 thread function
    def server1_thread(self):
        try:
            server1_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server1_socket.bind((self.SERVER1_HOST, self.SERVER1_PORT))
            server1_socket.listen(1)
            conn, addr = server1_socket.accept()
            print(f"Connected to {addr}")
            self.manage_client_server1(conn)
        except KeyboardInterrupt:
            print("Server 1 is shutting down...")
        finally:
            server1_socket.close()

    # Function to continuously send frames to connected clients
    def send_frames_to_clients(self):
        while True:
            frame_data = self.frame_queue.get()
            for client_socket in self.connected_clients:
                try:
                    client_socket.sendall(struct.pack("L", len(frame_data)) + frame_data)
                    print(f"Sent frame: {len(frame_data)} bytes")
                except Exception as e:
                    print(f"Error sending frame to client: {e}")

    # Server 2 thread function
    def server2_thread(self):
        try:
            server2_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server2_socket.bind((self.SERVER2_HOST, self.SERVER2_PORT))
            server2_socket.listen(3)  # Listen for up to 3 clients
            send_frames_thread = threading.Thread(target=self.send_frames_to_clients)
            send_frames_thread.start()
            print("Server 2 started. Waiting for clients...")
            while True:
                conn, addr = server2_socket.accept()
                print(f"Connected to {addr}")
                self.connected_clients.append(conn)
        except KeyboardInterrupt:
            print("Server 2 is shutting down...")
        finally:
            server2_socket.close()

    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('127.0.0.1', 7000))
        server_socket.listen(3)
        print("Server started. Waiting for clients...")

        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Connection established with {client_address}")
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()

    def handle_client(self, client_socket):
        while True:
            try:
                data = client_socket.recv(1024).decode()
                if not data:
                    break
                
                if data.startswith("a = "):
                    self.a = "True" in data.split(" = ")[1]
                if data.startswith("b = "):
                    self.b = "True" in data.split(" = ")[1]
                if data.startswith("c = "):
                    self.c = "True" in data.split(" = ")[1]
                if data.startswith("d = "):
                    self.d = "True" in data.split(" = ")[1]
                
                # Print current values of a, b, c, d
                print(f"Current values - a: {self.a}, b: {self.b}, c: {self.c}, d: {self.d}")

                # Update variables and UI
                self.communicate.update_signal.emit(self.a, self.b, self.c, self.d)

            except Exception as e:
                print(f"Error: {e}")
                break

        client_socket.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("Exiting...")
        # Close sockets and release ports here
        sys.exit(0)
