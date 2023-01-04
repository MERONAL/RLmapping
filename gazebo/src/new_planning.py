#!/usr/bin/env python
import os
import rospy

from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from gazebo.src.new_map import TopologicalMap


map = TopologicalMap(meta_radius=0.5, ifplot=True)
file_path = '/home/mhy/Desktop/research/navi/xue/src/gazebo/src/ampt_actloc.pkl'
if os.path.exists(file_path):
    map.load(file_path=file_path)
    map.plot()

def get_odim(odom_msg:Odometry):
    global cur_pos
    pos_x = odom_msg.pose.pose.position.x
    pos_y = odom_msg.pose.pose.position.y
    cur_pos = [pos_x, pos_y]
    map.new_node(pos_x, pos_y)
    

def get_global_path(data:PoseStamped):
    global cur_pos, path_planning_pub, prev_pos
    if cur_pos == None or cur_pos == prev_pos: return
    target_pos = [data.pose.position.x, data.pose.position.y]
    pathes = map.path_planning(cur_pos, target_pos)
    pathes_ros = map.path2ros_path(pathes)

    path_planning_pub.publish(pathes_ros)
    prev_pos = cur_pos


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rospy.init_node('map_listener', anonymous=True)
    global cur_pos, path_planning_pub, prev_pos
    cur_pos = None
    prev_pos = None
    path_planning_pub = rospy.Publisher("/global_navigation_path", Path, queue_size=5)
    rospy.Subscriber("/sim_p3at/odom",Odometry, get_odim,queue_size=1)
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, get_global_path, queue_size=1)
    # give target 
    plt.show(block=True)
    # rospy.spin()
    map.save(file_path='/home/mhy/Desktop/research/navi/xue/src/gazebo/src/ampt_actloc.pkl')
    

