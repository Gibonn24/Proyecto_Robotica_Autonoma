#!/usr/bin/env python3
"""Detect beverage cans using a YOLO model and the UR5 camera."""

import os
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class CanDetector:
    def __init__(self):
        self.bridge = CvBridge()
        # Load model path from ROS parameter or use default next to this script
        script_dir = os.path.dirname(os.path.realpath(__file__))
        default_path = os.path.join(script_dir, 'best.pt')
        model_path = rospy.get_param('~model_path', default_path)

        if YOLO is None:
            rospy.logerr('ultralytics package not found. Install it with pip.')
            raise RuntimeError('ultralytics package not available')
        self.model = YOLO(model_path)

        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image,
                                          self.image_callback, queue_size=1,
                                          buff_size=2**24)

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run detection
        results = self.model(frame)

        # Draw detections on the frame
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                label = self.model.names[cls]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

        cv2.imshow('Can Detection', frame)
        cv2.waitKey(1)


def main():
    rospy.init_node('can_detector')
    detector = CanDetector()
    rospy.loginfo('Can detector running...')
    rospy.spin()


if __name__ == '__main__':
    main()
