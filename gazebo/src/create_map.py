#!/usr/bin/env python
# from gazebo.src.map_plot import Graph
from map import Graph
import rospy
from nav_msgs.msg import Odometry
from amr_robots_gazebo.srv import Planning, PlanningResponse

graph = Graph(radius=1)
graph.load_map(file_name='/home/mhy/Desktop/research/navi/xue/src/gazebo/src/big_scene_copy.map')
def get_odim(odom_msg:Odometry):
    global cur_pos
    pos_x = odom_msg.pose.pose.position.x
    pos_y = odom_msg.pose.pose.position.y
    cur_pos = [pos_x, pos_y]
    graph.add_node(pos_x, pos_y)

    

def handle_cb(req):
    if cur_pos == None:
        return PlanningResponse()
    global_target = [req.x, req.y]
    local_target = graph.path_planing(cur_pos, global_target)
    res = PlanningResponse()
    res.x = local_target[0]
    res.y = local_target[1]
    return res


def handle_explore(req):
    if abs(req.x-100) < 1e-2:
        graph.save_map("/home/mhy/Desktop/research/navi/xue/src/gazebo/src/created_by_explore", save_fig=True)
    if cur_pos == None:
        return PlanningResponse()
    local_target = graph.explore(cur_pos)
    res = PlanningResponse()
    res.x = local_target[0]
    res.y = local_target[1]
    return res

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    rospy.init_node('planning_server', anonymous=True)
    global cur_pos,local_target_pub
    cur_pos = None
    rospy.Subscriber("/sim_p3at/odom",Odometry, get_odim,queue_size=1)
    rospy.Service('/local_target',Planning, handle_cb)
    # rospy.Service('explore',Planning, handle_explore)
    # give target 
    # plt.show(block=True)
    rospy.spin()
    graph.save_map(file_name='big_scene.map',save_fig=True)
    

