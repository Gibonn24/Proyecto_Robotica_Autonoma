#!/usr/bin/env python3
"""Scan three shelves with the UR5 camera and record can positions."""

import os
import csv
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class ShelfScanner:
    def __init__(self):
        self.bridge = CvBridge()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        default_model = os.path.join(script_dir, 'best.pt')
        model_path = rospy.get_param('~model_path', default_model)

        if YOLO is None:
            rospy.logerr('ultralytics package not found. Install it with pip.')
            raise RuntimeError('ultralytics package not available')
        self.model = YOLO(model_path)

        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image,
                                          self.color_callback, queue_size=1,
                                          buff_size=2**24)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image,
                                          self.depth_callback, queue_size=1,
                                          buff_size=2**24)
        self.info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo,
                                         self.info_callback, queue_size=1)

        self.color_img = None
        self.depth_img = None
        self.K = None
        self.detections = []  # (brand, (x, y, z))

        self.joint_pub = rospy.Publisher('/trajectory_controller/command',
                                         JointTrajectory, queue_size=10)
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
                            'elbow_joint', 'wrist_1_joint',
                            'wrist_2_joint', 'wrist_3_joint']

        self.shelf_positions = rospy.get_param('~shelf_positions', [
            [0.0, -1.4, 1.2, -1.8, -1.5, 0.0],  # bottom shelf
            [0.0, -1.1, 1.4, -1.8, -1.5, 0.0],  # middle shelf
            [0.0, -0.8, 1.6, -1.8, -1.5, 0.0],  # top shelf
        ])

    def color_callback(self, msg):
        self.color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def info_callback(self, msg):
        if self.K is None:
            self.K = msg.K
            self.info_sub.unregister()

    def move_arm(self, positions):
        traj = JointTrajectory()
        traj.header = Header()
        traj.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = rospy.Duration(2.0)
        traj.points.append(pt)
        traj.header.stamp = rospy.Time.now()
        self.joint_pub.publish(traj)

    def process_frame(self):
        if self.color_img is None or self.depth_img is None or self.K is None:
            return
        results = self.model(self.color_img)
        fx, fy = self.K[0], self.K[4]
        cx, cy = self.K[2], self.K[5]
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)
                depth = float(self.depth_img[v, u])
                if depth == 0.0:
                    continue
                depth_m = depth / 1000.0  # convert mm to meters
                X = (u - cx) * depth_m / fx
                Y = (v - cy) * depth_m / fy
                Z = depth_m
                brand = self.model.names[int(box.cls)]
                self.detections.append((brand, (X, Y, Z)))

    def scan(self):
        rate = rospy.Rate(1)
        for pos in self.shelf_positions:
            self.move_arm(pos)
            rospy.sleep(3.0)
            self.process_frame()
            rate.sleep()

        if not self.detections:
            rospy.logwarn('No cans detected.')
            return

        out_path = os.path.join(rospy.get_param('~output_dir', '/tmp'),
                                'detected_cans.csv')
        with open(out_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['brand', 'x', 'y', 'z'])
            for brand, (x, y, z) in self.detections:
                writer.writerow([brand, f'{x:.3f}', f'{y:.3f}', f'{z:.3f}'])
        rospy.loginfo('Results saved to %s', out_path)


def main():
    rospy.init_node('scan_shelves')
    scanner = ShelfScanner()
    rospy.sleep(1.0)
    scanner.scan()


if __name__ == '__main__':
    main()
