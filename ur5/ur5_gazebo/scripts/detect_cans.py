#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import os

class CanDetector:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        if not os.path.isfile(model_path):
            rospy.logerr('Model file {} not found'.format(model_path))
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='local')
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/camera/color/image_raw', Image, self.callback, queue_size=1)

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr('Error converting image: {}'.format(e))
            return
        results = self.model(cv_image)
        annotated = results.render()[0]
        cv2.imshow('Detections', annotated)
        cv2.waitKey(1)

def main():
    rospy.init_node('can_detector')
    CanDetector()
    rospy.loginfo('Can detector node started')
    rospy.spin()

if __name__ == '__main__':
    main()
