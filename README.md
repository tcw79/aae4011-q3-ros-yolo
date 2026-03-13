
# AAE4011 Assignment 1 — Q3: ROS‑Based Vehicle Detection from Rosbag

**Student Name:** Tang Chung Wang  
**Student ID:** 24020073d  
**Date:** 13/3/2026

---

## 1. Overview

This project is about a ROS‑based pipeline that reads images from a rosbag, runs a YOLO object detector, and publishes vehicle detections for visualization. 

## 2. Detection Method

I used a YOLO‑based detector ( YOLOv8n ) running as a ROS node that subscribes to the camera image topic from the rosbag.  
The model was chosen because pretrained weights are available for general road scenes, it runs in real time on a GPU/CPU suitable for student laptops, and it has good accuracy for cars, buses and trucks. It is also easier to understand. 

The node performs the following steps:

- Subscribes to the image topic (e.g. `/camera/image_raw`) and converts ROS images to OpenCV format.  
- Runs YOLO inference on each frame to obtain bounding boxes and class scores.  
- Filters detections to **vehicle‑related** classes (car, truck, bus, motorcycle) and publishes annotated images / detection messages on a ROS topic for visualization.

## 3. Repository Structure

The important files and folders in this repository are:

- `src/` — ROS node for YOLO‑based detection from rosbag images (Python script).  
- `launch/` — Launch files to start the detector node and any visualization nodes (e.g. `rviz`).  
- `scripts/` — Helper scripts for running rosbag, converting topics, or preparing the model.  
- `CMakeLists.txt` and `package.xml` — ROS build and dependency configuration.  
- `README.md` — This document describes how to run and evaluate the project.

## 4. Prerequisites

- **OS:** Ubuntu under WSL2 on Windows 11.  
- **ROS:** [e.g. ROS Noetic] with a catkin workspace `~/catkin_ws`.  
- **Python:** [e.g. Python 3.8] with the following packages:
  - `torch`, `torchvision` (for YOLO model)  
  - `opencv-python`  
  - `numpy`  
  - Any YOLO library you used (e.g. `ultralytics`).
 
## 5. How to Run (Q3.1 — 2 marks)

1. **Clone the repository**

   Open a terminal in WSL and clone the ROS package into your catkin workspace (if not already cloned):

   ```bash
   cd ~/catkin_ws/src
   git clone https://github.com/tcw79/aae4011-q3-ros-yolo.git
   cd ..
   catkin_make

2. **Install dependencies**
   Make sure ROS is installed and your workspace can find the package:

   ```bash
   cd ~/catkin_ws
   source devel/setup.bash
   pip install torch torchvision opencv-python

3. **Build the ROS package**
   From the workspace root, build the package (run again after any code changes):

   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash

4. **Place the rosbag file**
   Copy the driving sequence rosbag to a known location. In this project, the bag is stored on the Windows D: drive and accessed from WSL:

   ```bash
   # Example path used in this assignment
   rosbag play /mnt/d/AAE4011/"Video 1"/2026-02-02-17-57-27.bag

5. **Launch the pipeline**
Terminal 1: start the ROS master
Terminal 2: launch the YOLO detection node from this package
Terminal 3: play the rosbag so frames are published to the image topic
   Use three terminals:

   ```bash
   roscore
   
   cd ~/catkin_ws
   source devel/setup.bash
   roslaunch aae4011_yolo yolo_from_bag.launch
   
   rosbag play /mnt/d/AAE4011/"Video 1"/2026-02-02-17-57-27.bag

## 6. Sample Results

- **Image extraction summary**

  The rosbag contains a short driving sequence with front‑view road images recorded from the vehicle camera.  
  Frames are read from the camera image topic and passed into the `aae4011_yolo` node, which runs YOLO on each frame as the bag plays in real time.  

- **Detection results**

  During the sequence, the detector identifies multiple vehicles (cars, buses, and trucks) in front of the camera.  
  This ROS code generates some bounding boxes around the vehicles with class labels and confidence scores.  
  A small number of vehicles may occasionally be missed due to distance and small pixels.
  A small number of vehicles may be identified wrongly as a bus identified as a truck.

---

## 7. Video Demonstration (Q3.2 — 5 marks)

**Video link:** (https://www.youtube.com/watch?v=v9LOwcJfkjk)

The 1–3 minute video demonstrates the full pipeline as follows:

1. Starting `roscore` in Terminal 1.  
2. Launching the `aae4011_yolo` package with `yolo_from_bag.launch` in Terminal 2.  
3. Playing the driving rosbag from the D: drive in Terminal 3.  
4. Showing the image viewer window where YOLO detections (bounding boxes and labels) appear on top of the video frames.  

---

## 8. Reflection & Critical Analysis 

### (a) What Did You Learn? 

Through this task, I learned how to integrate a deep learning detector into ROS and run it on recorded rosbag data. There were a lot of errors that appeared when I was inputting any codes that were useful, such as missed downloads, missed files and more.
I became more familiar with setting up a catkin workspace, adding a custom package, and using multiple terminals to manage `roscore`, rosbag playback, and the detection node.  
I also gained practical experience with Git and GitHub, including making and editing a repo, configuring remotes, resolving non‑fast‑forward push errors with `git fetch` and `git rebase`, and using personal access tokens for authentication.

### (b) How Did You Use AI Tools? 

I used AI assistants (mainly Perplexity) to help with environment setup, Git problems, and documentation.  
It guided me when I could not push to GitHub, suggested specific commands such as `git remote set-url`, `git fetch origin`, and `git rebase origin/main`, and helped me fix the errors during the ROS coding.  
AI support saved time when I forgot the exact ROS or Git syntax, but I still had to run all commands, debug errors in my own terminal, and check that the final behaviour of the node and rosbag playback matched the expectations.  

### (c) How to Improve Accuracy? 

To improve detection accuracy, I think it needs to fine‑tune the YOLO model on data. For example, by labelling additional frames with vehicles in similar lighting and traffic conditions, to reduce the chances of missing vehicles.  
A second improvement is to adjust inference settings, such as using a larger input resolution or a higher‑capacity model, and carefully tuning confidence and non‑maximum‑suppression thresholds so that obvious false positives are filtered out.

### (d) Real‑World Challenges 

A YOLO model may not reach real‑time frame rates without optimisation or hardware accelerators.  
Another challenge is consistency: real‑world driving introduces motion blur, changing illumination, weather effects, and more, so the detector must remain reliable under conditions that are clean rosbag recording.  
In addition, it may require careful handling of false positives and missed detections to avoid unsafe behaviour.


---

## 9. References

1. Ultralytics. (2022). Ultralytics YOLO [Computer software]. GitHub. 
https://github.com/ultralytics/ultralytics
​
2. Ultralytics. (2023). Ultralytics YOLO documentation [Documentation]. Ultralytics. 
https://docs.ultralytics.com
​
3. ROS Wiki. (2020). image_transport [Documentation]. Open Source Robotics Foundation. 
http://wiki.ros.org/image_transport
​
4. ROBOMECHTRIX. (2021, February 6). Rosbags | ROS tutorial for beginners [Video]. YouTube. 
https://www.youtube.com/watch?v=Vlp0e89TXpI
 
​5. The Construct. (2022, November 17). What is rosbag? How to record and playback ROS topics [Blog post]. The Construct Sim. 
https://www.theconstruct.ai/ros-5-mins-045-rosbag-record-playback-ros-topics/
​
6. GitHub. (n.d.). Managing your personal access tokens [Documentation]. GitHub Docs. 
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
​
7. GitHub. (2024). Dealing with non-fast-forward errors [Documentation]. GitHub Docs. 
https://docs.github.com/en/get-started/using-git/dealing-with-non-fast-forward-errors
​


  
   






  
