# ROS2_DRL_Navigation
Implementation and comparison of DRL methods (PPO, SAC, and DDPG) in obstacle-free robot navigation.

## Table of Contents

## Project Structure
Current placeholder project structure
```txt
.
├── 📂 .devcontainer/: handle Dev Container creation in VSCode
│   ├── 📄 devcontainer.json - 
│   └── 📄 Dockerfile - includes the Docker commands for package installation and regular user setup
├── 📂 python/: repository for Python implementation (training and simulation) of DRL algorithms
│   ├── 📂 DDPG/: files related to DDPG training and testing
│   ├── 📂 PPO/: files related to PPO training and testing
│   └── 📂 SAC/: files related to SAC training and testing
├── 📂 ros2_ws/src/: contains the ROS2 packages for DRL integration
│   ├── 📂 control/: 
│   ├── 📂 description/: launch files for agent description
│   └── 📂 navDRL/: 
└── 📂 (planned) models/: contains the trained models for easy access from either the python or ros2_ws environments 
```

## Requirements
