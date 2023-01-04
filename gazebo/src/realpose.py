#!/usr/bin/env python
import os
import sys, time
import cv2
import numpy as np
import pylab as plt
import rospy
import math
from tf import TransformListener

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseWithCovarianceStamped
#from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

robotname = 'pioneer3at_robot'

def realposecb(data):
    msg = Odometry()
    msg.header.stamp = rospy.get_rostime()
    #rospy.loginfo('get {},{},{}'.format(data.name[-1], data.pose[-1], data.twist[-1]))
    global realposepub, realtwistpub
    i = len(data.name) - 1
    while(i >= 0):
        if(data.name[i] == robotname):
            msg.pose.pose = data.pose[i]
            msg.twist.twist = data.twist[i]
            # msgpose = Pose()
            # msgpose = data.pose[i]
            realposepub.publish(msg)
            # msgtwist = Twist()
            # msgtwist = data.twist[i]
            # realtwistpub.publish(msgtwist)
            break
        i -= 1

if __name__ == '__main__':
    rospy.init_node('realposepub', anonymous=True)
    rospy.Subscriber("/gazebo/model_states", ModelStates, realposecb, queue_size=1)
    global realposepub, realtwistpub
    realposepub = rospy.Publisher('/gt_posetwist', Odometry, queue_size=1)
    posecovarpub = rospy.Publisher('/odom/pose', PoseWithCovarianceStamped, queue_size=1)
    #realtwistpub = rospy.Publisher('/gt_twist', Twist, queue_size=1)
    rate = rospy.Rate(10)
    pose_msg = PoseWithCovarianceStamped()
    pose_msg.header.frame_id = 'odom'
    tfl = TransformListener()
    rospy.loginfo('real pose pub init done')
    while not rospy.is_shutdown():
        if tfl.frameExists("base_link"):
            # and tfl.frameExists("map"):
            try:
                t = tfl.getLatestCommonTime("base_link", "odom")
                pos, quat = tfl.lookupTransform("odom", "base_link", t)
                pose_msg.header.stamp = t;
                pose_msg.pose.pose.position.x = pos[0];
                pose_msg.pose.pose.position.y = pos[1];
                pose_msg.pose.pose.position.z = pos[2];
                pose_msg.pose.pose.orientation.x = quat[0];
                pose_msg.pose.pose.orientation.y = quat[1];
                pose_msg.pose.pose.orientation.z = quat[2];
                pose_msg.pose.pose.orientation.w = quat[3];
                posecovarpub.publish(pose_msg)
            except:
                pass
        rate.sleep()
    rospy.loginfo('shutdown')