FROM osrf/ros:humble-desktop-full

# Install system tools
RUN apt-get update && apt-get install -y \
  # iproute2 and iputils-ping to perform "ip addr" and "ping", 
  iproute2 \
  iputils-ping \
  # joystick mapping relay from robot to this container on the dev machine
  joystick \
  nano \
&& rm -rf /var/lib/apt/lists/*

# Install ROS2 nodes/libraries
RUN apt-get update && apt-get install -y \
    # test the ROS communication between the robot's container and dev machine's container
    ros-${ROS_DISTRO}-demo-nodes-cpp \
    ros-${ROS_DISTRO}-demo-nodes-py \
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-ros-gz* \
    ros-${ROS_DISTRO}-gz-ros2-control \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    # ros-${ROS_DISTRO}-turtlesim \
    ros-${ROS_DISTRO}-slam-toolbox \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-joy \
    ros-${ROS_DISTRO}-joy-teleop \
    ros-${ROS_DISTRO}-tf-transformations \
    ros-${ROS_DISTRO}-gazebo-ros \
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    # ros-${ROS_DISTRO}-gazebo-ros2-control \
    ros-${ROS_DISTRO}-plotjuggler \
    ros-${ROS_DISTRO}-plotjuggler-ros \
    # Handle Twist/TwistStamped messages
    ros-${ROS_DISTRO}-twist-mux \
    ros-${ROS_DISTRO}-twist-stamper \
    # Nav2 stack
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-nav2-bringup \
    ros-${ROS_DISTRO}-rqt-tf-tree  && \
    rm -rf /var/lib/apt/lists/*

# Add ubuntu user with same UID and GID as your host system, if it doesn't already exist
# Since Ubuntu 24.04, a non-root user is created by default with the name vscode and UID=1000
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi
# Add sudo support for the non-root user
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Add user to video group to allow access to webcam
RUN usermod --append --groups video $USERNAME

# Add user to render group for Ignition to access moden EGL/render nodes
RUN groupadd -g 110 render || true \
    && usermod --append --groups render $USERNAME

# Switch from root to user
USER $USERNAME

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install Git
RUN sudo apt install -y git

# Rosdep update
RUN rosdep update

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# Install python 3 and pip
RUN sudo apt-get install -y python3 python3-pip && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Install necessary pip packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gymnasium ipykernel

################################
## ADD ANY CUSTOM SETUP BELOW ##
################################
