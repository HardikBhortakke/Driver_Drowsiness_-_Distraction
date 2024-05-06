import sys
import socket
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from ess import Ui_MainWindow  # Importing the generated UI class


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
        self.timer.start(10)  # Update every 10 milliseconds

        # Start server to listen to clients
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.start()

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

    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("localhost", 9999))
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
    sys.exit(app.exec_())
