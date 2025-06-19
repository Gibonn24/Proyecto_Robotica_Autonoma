#!/usr/bin/env python3
import rospy
import numpy as np
from visualization_msgs.msg import Marker


# ------------------------ CONFIG ------------------------ #

DEFAULT_FRAME = "world"  # En lugar de base_link si usas Gazebo
BALL_TOPIC = "ball_marker_topic"
FRAME_TOPIC = "frame_marker_topic"

# --------------------- COLORES -------------------------- #

color = {
    'RED': (1.0, 0.0, 0.0),
    'GREEN': (0.0, 1.0, 0.0),
    'BLUE': (0.0, 0.0, 1.0),
    'YELLOW': (1.0, 1.0, 0.0),
    'PINK': (1.0, 0.0, 1.0),
    'CYAN': (0.0, 1.0, 1.0),
    'BLACK': (0.0, 0.0, 0.0),
    'DARKGRAY': (0.2, 0.2, 0.2),
    'LIGHTGRAY': (0.5, 0.5, 0.5),
    'WHITE': (1.0, 1.0, 1.0)
}

# ---------------------- BALL MARKER --------------------- #

class BallMarker:
    id = 0

    def __init__(self, color_val, alpha=1.0, scale=0.05):
        frame = rospy.get_param("~reference_frame", DEFAULT_FRAME)
        self.pub = rospy.Publisher(BALL_TOPIC, Marker, queue_size=10)
        self.marker = Marker()
        self.marker.header.frame_id = frame
        self.marker.ns = "ball_marker"
        self.marker.id = BallMarker.id
        BallMarker.id += 1
        self.marker.type = Marker.SPHERE
        self.marker.action = Marker.ADD
        self.marker.pose.orientation.w = 1.0
        self.marker.scale.x = self.marker.scale.y = self.marker.scale.z = scale
        self.set_color(color_val, alpha)
        self.marker.lifetime = rospy.Duration()

    def set_color(self, color_val, alpha=1.0):
        self.marker.color.r, self.marker.color.g, self.marker.color.b = color_val
        self.marker.color.a = alpha

    def position(self, T):
        self.marker.pose.position.x = T[0, 3]
        self.marker.pose.position.y = T[1, 3]
        self.marker.pose.position.z = T[2, 3]
        self.publish()

    def xyz(self, position):
        self.marker.pose.position.x, self.marker.pose.position.y, self.marker.pose.position.z = position
        self.publish()

    def publish(self):
        self.marker.header.stamp = rospy.Time.now()
        self.pub.publish(self.marker)

# -------------------- FRAME MARKER ---------------------- #

class FrameMarker:
    id = 0

    def __init__(self, color_saturation=1.0, alpha=1.0, scale=0.1):
        frame = rospy.get_param("~reference_frame", DEFAULT_FRAME)
        self.pub = rospy.Publisher(FRAME_TOPIC, Marker, queue_size=10)
        self.markers = []

        for i, axis in enumerate(["x", "y", "z"]):
            marker = Marker()
            marker.header.frame_id = frame
            marker.ns = "frame_markers"
            marker.id = FrameMarker.id
            FrameMarker.id += 1
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = scale
            marker.scale.y = marker.scale.z = 0.01
            marker.color.a = alpha
            if axis == "x":
                marker.color.r = color_saturation
            elif axis == "y":
                marker.color.g = color_saturation
                q = quaternion_mult([1, 0, 0, 0], [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
                marker.pose.orientation.w, marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z = q
            else:  # z
                marker.color.b = color_saturation
                q = quaternion_mult([1, 0, 0, 0], [np.cos(-np.pi/4), 0, np.sin(-np.pi/4), 0])
                marker.pose.orientation.w, marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z = q
            marker.pose.orientation.w = marker.pose.orientation.w if axis == "x" else marker.pose.orientation.w
            self.markers.append(marker)

    def set_pose(self, pose):
        for marker in self.markers:
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = pose[:3]
        self.publish()

    def publish(self):
        for marker in self.markers:
            marker.header.stamp = rospy.Time.now()
            self.pub.publish(marker)

# ------------------ QUAT UTIL --------------------------- #

def quaternion_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]

# -------------------- TEST NODE ------------------------- #

if __name__ == "__main__":
    rospy.init_node("marker_test_node")
    ball = BallMarker(color['GREEN'])
    ball.xyz([0.5, 0.0, 0.5])

    frame = FrameMarker()
    frame.set_pose([0.5, 0.0, 0.5])

    rospy.spin()
