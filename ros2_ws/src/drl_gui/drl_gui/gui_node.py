# import packages:
import sys
import os
import time
import threading
import signal
import subprocess
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox, QPushButton
from PyQt5.QtCore import QTimer, Qt, pyqtSignal

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
    # signal for updating GUI from background threads:
    navigation_finished = pyqtSignal()
    resetting_finished = pyqtSignal()

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
        grid1.addWidget(QLabel("X"),         0, 0, alignment = Qt.AlignCenter)  # labelled text widget for x position
        grid1.addWidget(QLabel("Y"),         0, 1, alignment = Qt.AlignCenter)  # labelled text widget for y position
        grid1.addWidget(QLabel("Tolerance"), 0, 2, alignment = Qt.AlignCenter)  # labelled text widget for tolerance

        self.x_input = QLineEdit()          # instantiate the text input for x position, so that it may be used later
        self.y_input = QLineEdit()          # instantiate the text input for y position, so that it may be used later
        self.tolerance_input = QLineEdit()  # instantiate the text input for tolerance, so that it may be used later

        grid1.addWidget(self.x_input,         1, 0)     # add the x position to the gui
        grid1.addWidget(self.y_input,         1, 1)     # add the y position to the gui
        grid1.addWidget(self.tolerance_input, 1, 2)     # add the tolerance to the gui

        # grid 2 - model selection:
        grid2.addWidget(QLabel("Model"),  0, 0, alignment = Qt.AlignCenter)     # add the label to the grid
        self.combo_box = QComboBox()                                            # instantiate the ComboBox
        policy_path = get_package_share_directory("drl_policy")                 # define the path to the policy which contains the models
        model_path = os.path.join(policy_path, "policy")                        # get the path of the model directory
        models = os.listdir(model_path)                                         # list out the models within this directory

        # for every model in the directory:
        for model in models:
            # add the model to the ComboBox:
            self.combo_box.addItem(model)

        # add the ComboBox to the grid:
        grid2.addWidget(self.combo_box, 1, 0, alignment = Qt.AlignCenter) 

        # row 1 - buttons:
        self.nav_button = QPushButton("Navigate")                       # instantiate the QPushButton
        self.nav_button.clicked.connect(self._on_nav_button_clicked)    # connect its functionality  
        row1.addWidget(self.nav_button)                                 # add to gui

        self.reset_button = QPushButton("Reset")                            # instantiate the QPushButton
        self.reset_button.clicked.connect(self._on_reset_button_clicked)    # connect its functionality  
        row1.addWidget(self.reset_button)                                   # add to gui

        # add the grids/rows to the outer layout:
        main_layout.addLayout(grid1)
        main_layout.addLayout(grid2)
        main_layout.addLayout(row1)

        # apply the layout:
        self.setLayout(main_layout)

        # connect signals:
        self.navigation_finished.connect(self._on_navigation_finished)
        self.resetting_finished.connect(self._on_reset_finished)

    # function that the nav button is to execute:
    def _on_nav_button_clicked(self):
        # I think that this navigation button should launch both the policy node and the goal client.
        # - it should append the model type to the policy node
        # - it should pull values from the class to populate the goal client
        # - it should listen to the goal client and its feedback, and if the navigation either fails or was successful,
        #   it should shut down both the goal client and the policy node.
        
        # lock user out while button is running:
        self.nav_button.setEnabled(False)
        self.nav_button.setText("Running...")

        # gather the values needed to launch:
        model_name = self.combo_box.currentText()
        x = self.x_input.text()
        y = self.y_input.text()
        tolerance = self.tolerance_input.text()

        if not x or not y or not tolerance:
            print("Please pass a value for navigation.")
            self._on_navigation_finished()
        else:
            # launch the policy node:
            self.policy_process = subprocess.Popen(["ros2", "run", "drl_policy", "policy_node",
                                                    "--ros-args", "-p", f"model_name:={model_name}"], 
                                                    start_new_session = True
                                                    )
            
            # launch the goal client:
            self.goal_process = subprocess.Popen(["ros2", "run", "drl_policy", "goal_client", x, y, tolerance])

            # monitor in a thread so the GUI does not kill itself:
            threading.Thread(target = self._monitor_process, daemon = True).start()

    # function that the reset button is to execute:
    def _on_reset_button_clicked(self):
        # lock user out while button is running:
        self.reset_button.setEnabled(False)
        self.reset_button.setText("Resetting...")

        # use another thread:
        threading.Thread(target = self._reset_process, daemon = True).start()

    # method for monitoring if navigation is done:
    def _monitor_process(self):
        # wait for the goal process to finish, then kill nodes:
        self.goal_process.wait()
        os.killpg(os.getpgid(self.policy_process.pid), signal.SIGTERM)
        self.policy_process.wait()

        # re-enable the button from background thread safely using a signal:
        self.navigation_finished.emit()

    # method for doing the reset process:
    def _reset_process(self):
        # define initial position of the agent:
        x = "-3.0"
        y = "-3.0"
        z = "0.0"

        # move the position of the agent:
        subprocess.run(["ign", "service", "-s", "/world/world_1/set_pose",
                        "--reqtype", "ignition.msgs.Pose",
                        "--reptype", "ignition.msgs.Boolean", 
                        "--timeout", "2000",
                        "--req", f"name: 'agent', position: {{x: {x}, y: {y}, z: {z}}}"])

        # kill the previous nodes:
        subprocess.run(["pkill", "-f", "rf2o_laser_odometry"])
        subprocess.run(["pkill", "-f", "covariance_filter"])
        subprocess.run(["pkill", "-f", "ekf_filter_node"])

        # call the launch file:
        subprocess.Popen(["ros2", "launch", "agent_bringup", "launch_odom.py"])

        # re-enable the button:
        time.sleep(2)
        self.resetting_finished.emit()

    # method for re-enabling the nav button:
    def _on_navigation_finished(self):
        # modify the button state:
        self.nav_button.setEnabled(True)
        self.nav_button.setText("Navigate")

    # method for re-enabling the reset button:
    def _on_reset_finished(self):
        # modify the button state:
        self.reset_button.setEnabled(True)
        self.reset_button.setText("Reset")
        
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


