#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class YoloRosNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.delay = 1  # playback delay control

        # Change this path if your username/folder is different
        model_path = rospy.get_param("~model_path", "/home/user/models/yolov8n.pt")
        self.model = YOLO(model_path)

        image_topic = rospy.get_param("~image_topic", "/hikcamera/image_2/compressed")

        self.sub = rospy.Subscriber(
            image_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1
        )

        rospy.loginfo("YoloRosNode initialized. Subscribing to %s", image_topic)
        rospy.loginfo("Using model at %s", model_path)

    def image_callback(self, msg):
        # Decode compressed image to OpenCV frame
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            rospy.logwarn("Failed to decode image")
            return

        # Run YOLO inference
        results = self.model(frame)[0]

        # Draw detections (vehicles only)
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = self.model.names[int(cls)]

            if label not in ["car", "truck", "bus", "motorbike", "motorcycle"]:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,              # larger font
                (0, 255, 0),
                2                 # thicker text
            )

        # Optional on-screen help
        cv2.putText(
            frame,
            "Keys: P=pause 1=normal 2=slow 3=fast Q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Show frame
        cv2.imshow("ROS YOLO Vehicle Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        # Playback controls
        if key == ord('p'):
            # Pause until any key is pressed
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 != 255:
                    break

        elif key == ord('1'):
            self.delay = 1  # normal speed

        elif key == ord('2'):
            self.delay = 50  # slower (extra delay)

        elif key == ord('3'):
            self.delay = 1   # fast (minimal extra delay)

        elif key == ord('q'):
            rospy.signal_shutdown("User requested quit")

        if self.delay > 1:
            cv2.waitKey(self.delay)

if __name__ == "__main__":
    rospy.init_node("aae4011_yolo_node")
    node = YoloRosNode()
    rospy.loginfo("YOLO ROS node started.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
