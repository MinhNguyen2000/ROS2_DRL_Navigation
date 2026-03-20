# import packages:
import sys
import threading
import signal
import rclpy
from rclpy.node import Node
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import QTimer

# node class:
class GuiNode(Node):
    # constructor for node:
    def __init__(self):
        # inherit from parent type:
        super().__init__("gui_node")
        # display to user on startup:
        self.get_logger().info("GUI node started")

# this is a class which serves as the WINDOW:
class MainWindow(QWidget):
    # constructor for window:
    def __init__(self):
        # inherit from parent type:
        super().__init__()

        # use internal method of QWidget to set the title of the window:
        self.setWindowTitle("ROS2 DRL GUI")

        # this is a layout manager, which lives inside the widget (QWidget):
        layout = QVBoxLayout()

        # this is a child widget:
        self.input_field = QLineEdit() 
        self.input_field.setPlaceholderText("enter into this field")

        # add the child widget to the layout manager, along with a label:
        layout.addWidget(QLabel("input: ")) 
        layout.addWidget(self.input_field)

        # add a button: 
        self.button = QPushButton("button") # add a child widget
        self.button.clicked.connect(self._on_button_clicked)    # add a functionality to the button, which is defined as a function
        layout.addWidget(self.button)   # add the parent to the child

        # apply the layout:
        self.setLayout(layout)

    # function that the button is to execute:
    def _on_button_clicked(self):
        text = self.input_field.text()
        print(f"button clicked, input was: {text}") 
    
# define the main execution of the node:
def main():
    # initialize rclpy:
    rclpy.init()

    # instantiate the node:
    node = GuiNode()

    # spin ROS2 in a background thread so it doesn't block the GUI from working:
    ros_thread = threading.Thread(target = rclpy.spin, args = (node, ), daemon = True)
    ros_thread.start()

    # start the GUI:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # allow python to read signals every 500ms:
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda:None)

    # handle shutdown:
    signal.signal(signal.SIGINT, lambda *args: app.quit())
    exit_code = app.exec_()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)

# main:
if __name__ == "__main__":
    main()


