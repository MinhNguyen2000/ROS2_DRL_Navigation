# import packages:
import sys
import os
import threading
import signal
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox, QPushButton
from PyQt5.QtCore import QTimer, Qt

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
        main_layout = QVBoxLayout() # this will be the outer layout, stacking rows vertically
       
        # instantiate a series of child layouts:
        grid1 = QGridLayout()     # this is a 2,3 grid, used for the x, y, and tolerance inputs
        grid2 = QGridLayout()     # this is a 2,1 grid, used for selecting the model
        row1 = QHBoxLayout()      # this is a single row that will contain the navigation button

        # need to define what each row/grid contains now:
        # grid 1 - goal parameters:
        grid1.addWidget(QLabel("x"),         0, 0, alignment = Qt.AlignCenter)
        grid1.addWidget(QLabel("y"),         0, 1, alignment = Qt.AlignCenter)
        grid1.addWidget(QLabel("tolerance"), 0, 2, alignment = Qt.AlignCenter)

        self.x_input = QLineEdit()
        self.y_input = QLineEdit()
        self.tolerance_input = QLineEdit()

        grid1.addWidget(self.x_input,         1, 0)
        grid1.addWidget(self.y_input,         1, 1)
        grid1.addWidget(self.tolerance_input, 1, 2)

        # grid 2 - model selection:
        grid2.addWidget(QLabel("model"),  0, 0, alignment = Qt.AlignCenter)     # add the label to the grid
        combo_box = QComboBox()                                                 # instantiate the ComboBox
        policy_path = get_package_share_directory("drl_policy")                 # define the path to the policy which contains the models
        model_path = os.path.join(policy_path, "policy")                        # get the path of the model directory
        models = os.listdir(model_path)                                         # list out the models within this directory

        # for every model in the directory:
        for model in models:
            # add the model to the ComboBox:
            combo_box.addItem(model)

        # add the ComboBox to the grid:
        grid2.addWidget(combo_box, 1, 0, alignment = Qt.AlignCenter)

        # row 1 - button for navigation:
        nav_button = QPushButton("navigate!")
        nav_button.clicked.connect(self._on_button_clicked)
        row1.addWidget(nav_button)

        # add the grids/rows to the outer layout:
        main_layout.addLayout(grid1)
        main_layout.addLayout(grid2)
        main_layout.addLayout(row1)

        # apply the layout:
        self.setLayout(main_layout)

    # function that the button is to execute:
    def _on_button_clicked(self):
        print(f"button was pressed!") 
    
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


