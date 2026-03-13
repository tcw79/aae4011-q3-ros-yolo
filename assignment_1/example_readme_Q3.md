# AAE4011 – Assignment 1 Q3: ROS-Based Vehicle Detection from Rosbag

This document describes my implementation, environment, and usage instructions for Q3 (ROS-based vehicle detection) using a ROS node and YOLOv8.

---

## 1. Environment and dependencies

- OS: Ubuntu (WSL on Windows)
- ROS: ROS Noetic
- Catkin workspace: `~/catkin_ws`
- Package for Q3: `aae4011_yolo`

### 1.1 ROS packages

- `aae4011_yolo` (this repository)
- `ros-noetic-image-transport-plugins` (for handling `sensor_msgs/CompressedImage` topics)

Install with:

```bash
sudo apt-get update
sudo apt-get install ros-noetic-image-transport-plugins
