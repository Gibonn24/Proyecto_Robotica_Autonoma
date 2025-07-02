#!/usr/bin/env python3
"""
ros_pcl.py — Nodo ROS que convierte mensajes PointCloud2 a formato PCL.

Este nodo se suscribe al tópico `/points` (nube de puntos de una cámara RGB-D),
convierte los datos a un objeto PCL compatible usando ros2pcl y muestra información básica.
"""

import rospy
from sensor_msgs.msg import PointCloud2
from pcl_helper import ros2pcl  


def callback_pointcloud(msg):
    rospy.loginfo_once("Recibida nube de puntos. Convirtiendo a PCL...")
    cloud = ros2pcl(msg)
    rospy.loginfo(f"Nube convertida: {cloud.size} puntos")


def main():
    rospy.init_node("ros_pcl_node")
    rospy.Subscriber("/camera/depth_registered/points", PointCloud2, callback_pointcloud, queue_size=1)
    rospy.loginfo("Nodo ros_pcl_node suscrito a /points")
    rospy.spin()


if __name__ == '__main__':
    main()
