#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List
import rospy
from gazebo_msgs.msg import ModelState
import numpy as np
from std_msgs.msg import Float32MultiArray


class Obstacle:
    """单个动态障碍物"""

    ROS_TOPIC_FREQUENCY = 20

    def __init__(
        self,
        obstacle_name: str,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        min_velocity: float,
        max_velocity: float,
    ) -> None:
        """单个障碍物实例化，给定障碍物的起点和终点，障碍物在两个点之间会做变速运动
        Args:
            obstacle_name (str): 障碍物名称，【必须和gazebo中的object id对应】
            start_pos (np.ndarray): 起始点位置坐标
            end_pos (np.ndarray): 终点位置作为
            min_velocity (float): 最小运动速度，单位m/s
            max_velocity (float): 最大运动速度，单位m/s
        """
        self.obstacle_name = obstacle_name
        self._init(start_pos, end_pos, min_velocity, max_velocity)

    def _init(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        min_velocity: float,
        max_velocity: float,
    ) -> None:
        """根据起始位置、终止位置、最小速度、最大速度计算障碍物移动‘密度’"""
        distance = np.linalg.norm(end_pos - start_pos)
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.density_min = self.ROS_TOPIC_FREQUENCY * distance / max_velocity
        self.density_max = self.ROS_TOPIC_FREQUENCY * distance / min_velocity
        # 求斜率
        k = (end_pos[1] - start_pos[1]) / (end_pos[0] - start_pos[0])
        self.get_y = lambda x: k * (x - start_pos[0]) + start_pos[1]
        self.right_direction = True
        self._density()

    def _density(self) -> None:
        """随机生成新范围"""
        density = np.random.randint(self.density_min, self.density_max)
        self.x = np.array(np.linspace(self.start_pos[0], self.end_pos[0], density))
        self.idx = np.random.randint(0, density)

    def update(self) -> ModelState:
        """更新障碍物位置, 并返回障碍物新位置信息"""
        if self.idx == self.x.shape[0] - 1:
            self.right_direction = False
            self._density()
            self.idx = self.x.shape[0] - 1
        if self.idx == 0:
            self.right_direction = True
            self._density()
            self.idx = 0

        if self.right_direction:
            self.idx += 1
        else:
            self.idx -= 1
        
        state = ModelState()
        state.model_name = self.obstacle_name
        state.pose.position.x = self.x[self.idx]
        state.pose.position.y = self.get_y(self.x[self.idx])
        return state


class DynamicObstaclePublisher:
    """动态障碍物发布器"""

    def __init__(self, obstacle_config:dict) -> None:
        """初始化障碍物发布器
        Args:
            obstacle_config (dict): 障碍物配置文件
        """
        Obstacle.ROS_TOPIC_FREQUENCY = obstacle_config["ROS_TOPIC_FREQUENCY"]
        self.obstacles = self._build_obstacles(obstacle_config['obstacles'])
        self._init()
    
    def _build_obstacles(self, obstacle_config:list) -> List[Obstacle]:
        """根据配置文件构建障碍物元组"""
        obstacles = []
        for obstacle in obstacle_config:
            tmp_obstacle=Obstacle(
                    obstacle['obstacle_name'],
                    np.array(obstacle['start_pos']),
                    np.array(obstacle['end_pos']),
                    obstacle['min_velocity'],
                    obstacle['max_velocity']
                    )
            obstacles.append(tmp_obstacle)
        return obstacles

    def _init(self) -> None:
        """初始化ROS节点、发布器"""
        rospy.init_node("dynamic_obstacle_publisher", anonymous=True)
        self.obstacle_pubs = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.obstacle_pose_pubs = rospy.Publisher("/dynamic_obstacle_pose", Float32MultiArray, queue_size=1)
        self.poses = [0 for _ in range(2 * len(self.obstacles))]

    def run(self) -> None:
        """开始发布障碍物位置信息"""
        rate = rospy.Rate(Obstacle.ROS_TOPIC_FREQUENCY)
        while not rospy.is_shutdown():
            for idx, obstacle in enumerate(self.obstacles):
                state = obstacle.update()
                self.obstacle_pubs.publish(state)
                self.poses[2 * idx] = state.pose.position.x
                self.poses[2 * idx + 1] = state.pose.position.y

            self.obstacle_pose_pubs.publish(Float32MultiArray(data=self.poses))
            rate.sleep()

# if __name__ == "__main__":
#     rospy.init_node("dynamic_obstacle_node", anonymous=True)
#     import sys
#     obstacle_config = eval(sys.argv[1])
#     dynamic_obstacle_publisher = DynamicObstaclePublisher(obstacle_config=obstacle_config)
#     dynamic_obstacle_publisher.run()


if __name__ == "__main__":
    import sys
    obstacle_config = eval(sys.argv[1])
    print('obstacle config:', obstacle_config)
    # print('-'*1000)
    #obstacle_config格式:
    # obstacle_config = {
    #     "ROS_TOPIC_FREQUENCY":10,
    #     "obstacles": [
    #     {
    #       "obstacle_name": "unit_cylinder_1",
    #       "start_pos": [-1, 5], # 动态障碍物运动起点坐标
    #       "end_pos": [4, 5], # 动态障碍物运动终点坐标
    #       "min_velocity": 0.1,
    #       "max_velocity": 0.2,
    #     },
    #     {
    #       "obstacle_name": "unit_cylinder_4",
    #       "start_pos": [2, 3], # 动态障碍物运动起点坐标
    #       "end_pos": [6, 3], # 动态障碍物运动终点坐标
    #       "min_velocity": 0.1,
    #       "max_velocity": 0.3,
    #     }
    #   ]
    # }
    dynamic_obstacle_publisher = DynamicObstaclePublisher(obstacle_config=obstacle_config)
    dynamic_obstacle_publisher.run()

