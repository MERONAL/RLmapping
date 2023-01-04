#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
import numpy as np

def laser_scan_callback(msg):
    data = msg.ranges
    data = np.array(data)
    print(np.min(data))

if __name__ == "__main__":
    rospy.init_node(name="mini_laser_scan", anonymous=True)
    rospy.Subscriber('/scan',LaserScan, laser_scan_callback, buff_size=1)
    rospy.spin()