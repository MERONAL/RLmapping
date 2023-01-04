#!/usr/bin/env python
from map_plot import Graph
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

graph = Graph(radius=1)
graph.load_map(file_name='/home/mhy/Desktop/research/navi/xue/src/gazebo/src/big_square_map.map')
def get_odim(odom_msg:Odometry):
    global cur_pos
    pos_x = odom_msg.pose.pose.position.x
    pos_y = odom_msg.pose.pose.position.y
    cur_pos = [pos_x, pos_y]
    graph.add_node(pos_x, pos_y)
    

def get_global_path(data:Point):
    global cur_pos, local_target_pub, prev_pos
    if cur_pos == None or cur_pos == prev_pos: return
    target_pos = [data.x, data.y]
    local_target = graph.path_planing(cur_pos, target_pos)
    tmp = Point()
    tmp.x = local_target[0]
    tmp.y = local_target[1]
    local_target_pub.publish(tmp)
    prev_pos = cur_pos


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rospy.init_node('map_listener', anonymous=True)
    global cur_pos,local_target_pub, prev_pos
    cur_pos = None
    prev_pos = None
    local_target_pub = rospy.Publisher("/navi_local_target", Point, queue_size=1)
    rospy.Subscriber("/sim_p3at/odom",Odometry, get_odim,queue_size=1)
    rospy.Subscriber("/navi_target", Point, get_global_path, queue_size=1)
    # give target 
    plt.show(block=True)
    # rospy.spin()
    graph.save_map(save_fig=True,file_name="big_square_map")
    

