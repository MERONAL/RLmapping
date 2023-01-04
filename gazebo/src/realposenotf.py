#!/usr/bin/env python

import os
import sys,time
import rospy
import math
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry

t_prior_visited = [ 17.5, 2.5, 0.0]

def realodomCB(data):
    global posecovarpub, posepub
    posecov_msg = PoseWithCovarianceStamped()
    pose_msg = PoseStamped()
    data.pose.pose.position.x = data.pose.pose.position.x - t_prior_visited[0]
    data.pose.pose.position.y = data.pose.pose.position.y - t_prior_visited[1]
    data.pose.pose.position.z = data.pose.pose.position.z - t_prior_visited[2]
    posecov_msg.pose = data.pose
    pose_msg.pose = data.pose.pose
    posecov_msg.header = data.header
    pose_msg.header = data.header

    posepub.publish(pose_msg)
    posecovarpub.publish(posecov_msg)

if __name__ == '__main__':
    rospy.init_node('realposepub', anonymous=True)
    global posecovarpub, posepub
    posepub = rospy.Publisher('/real/pose', PoseStamped, queue_size=1)
    posecovarpub = rospy.Publisher('/real/posecov', PoseWithCovarianceStamped, queue_size=1)
    rospy.Subscriber("/sim_p3at/odom", Odometry, realodomCB, queue_size=1)
    rate = rospy.Rate(10)
    rospy.loginfo('realpose pub start!')
    while not rospy.is_shutdown():
        rate.sleep()