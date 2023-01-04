#!/usr/bin/env python
import numpy as np
import rospy
import sys
import time
import os
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from std_msgs.msg import String
from std_msgs.msg import Float64
from std_msgs.msg import Int8MultiArray
from collections import Counter


def callback(map: OccupancyGrid):
    mapdata.data = map.data
    # pub = rospy.Publisher('mapprob', Int8MultiArray, queue_size=10)
    # pub.publish(mapdata)
    print(Counter(mapdata.data))
    explored_area = Counter(mapdata.data)[100] + Counter(mapdata.data)[0]
    print(explored_area)


def somethingCool():
    global mapdata
    mapdata = Int8MultiArray()
    rospy.init_node('test_name', anonymous=False)
    rospy.Subscriber("map", OccupancyGrid, callback)
    rospy.loginfo(mapdata)
    rospy.spin()


if __name__ == '__main__':
    try:
        somethingCool()
    except rospy.ROSInterruptException:
        pass
