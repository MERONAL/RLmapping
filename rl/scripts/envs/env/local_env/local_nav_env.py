"""
这个文件是用来做局部避障的环境，主要是用来做局部避障的训练
总共包含三个子环境：
    1. 离散动作空间的局部避障环境
        动作空间：0:前进、1:左转、2:右转
    2. 连续动作空间的局部避障环境
        动作空间：cmd_vel(2维), 策略网络需要输出2维度的[-1,1]的控制量，环境会将输出机器人的线速度被约束在[0, 0.3]之间,角速度被约束在[-0.3, 0.3]之间;
    3. 连续动作+速度增量的局部避障环境
        动作空间：cmd_vel(2维)，策略网络需要输出2维度的[-1,1]的控制量的【增量】delta v_t，环境会将输出机器人的线速度被约束在[0, 0.3]之间,角速度被约束在[-0.3, 0.3]之间;
        既真实的速度: v_t = v_{t-1} + 【delta v_t】
    4. 连续动作+速度增量+桢迭代解决POMDP的局部避障环境
        观测空间: 16 * num_stack_frames
"""

from typing import Tuple, List, Union
import time
from queue import deque

import rospy
import numpy as np
from gym import spaces

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gym.utils import seeding
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int8MultiArray
from collections import Counter

from rl.scripts.envs.gazebo_env import GazeboEnv
from rl.scripts.envs.utils import History, quaternion2euler
from rl.scripts.envs.assets.data.env_tools import CollisionChecker
from rl.scripts.envs.keys import (
    SUCCESS, FAILED, RUNNING,STATUS,
    POSITION,ORIENTATION,LocalConstants)


class LocalNavEnv(GazeboEnv):
    """
    static env for local obstacle avoidance.
    """

    def __init__(
            self,
            train_config: Union[dict, str],
            episode_maximum_steps=512,
            init_target_distance=2,
            target_distance_gap=10000,
            record_history=False,
            as_local_planner=False
    ) -> None:
        """局部导航环境类，提供基本的env step、reset、render等方法
        Args:
            train_config (dict|str): 训练时的config字典，存储一些路径和常量信息
            episode_maximum_steps (int, optional):  每个episode的最大步数. Defaults to 512.
            init_target_distance (int, optional): 初始化机器人的起点和目标点的【最大】距离. Defaults to 2.
            target_distance_gap (int, optional): 表示每get_target_distance_gap步，目标点和出发点的距离会增加1. Defaults to 10000.
            record_history (bool, optional): 是否记录轨迹信息. Defaults to False.
            as_local_planner (bool, optional): 是否作为局部避障器使用. Defaults to False.
        """

        super().__init__(train_config=train_config)
        # 检测是否将此工具作为局部避障器使用
        self._as_local_planner = as_local_planner
    
        # publisher: 机器人的速度控制指令发布者
        self.vel_pub = rospy.Publisher("/sim_p3at/cmd_vel", Twist, queue_size=5)
        # subscriber: 接收机器人的激光雷达数据和机器人的姿态数据订阅者
        self.scan_sub = rospy.Subscriber(
            "/scan", LaserScan, callback=self._scan_callback)
        self.pose_sub = rospy.Subscriber(
            "/sim_p3at/odom", Odometry, callback=self._odom_callback)
        self.map = rospy.Subscriber("map", OccupancyGrid, callback=self._map_callback)
        # 接收topic的信息
        self._odom_msg = None
        self._scan_data = None
        self._map_data = None


        # service proxy: 重置机器人的位置
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.set_gazebo = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        # 初始化碰撞检测模块
        self.collision_checker = CollisionChecker(scene_config=self.train_config['scene_config_path'],
                                                  scene=self.train_config['scene_name'])

        # 强化学习相关参数
        """
        _obs_dim:[pos_state, orientation, laser_data]，dim = 2 + 4 + 10 = 16
        pos_state: [x, y], dim=2, 机器人的位置
        orientation: [w, x, y, z], dim=4, 机器人的朝向
        laser_data: [laser_data], dim=10, 机器人的激光雷达数据（经过离散化处理）
        """
        self._obs_dim = 16
        self.reward_range = (-np.inf, np.inf)

        self.agent_current_pose = {
            POSITION: None,
            ORIENTATION: None 
        }
        self.agent_target_pose = {
            POSITION: None,# 当前目标点(临时生成)
            ORIENTATION: None # 当前目标点的朝向(临时生成)
        }
        self.explored_map_area = None
        self.total_map_area = 36781
        self.episode_maximum_steps = episode_maximum_steps  # 每个episode的最大步数
        self.current_step = 0  # 当前episode的步数, 用于判断是否超过最大步数，每次reset会置0
        self.total_step = 0  # 用来记录总的步数，根据这个步数来约束目标点和出发点的距离（课程学习）
        self.target_distance_gap = (
            target_distance_gap  # 表示每get_target_distance_gap步，目标点和出发点的距离会增加1.
        )

        # distance = total_step / target_distance_gap + self.init_target_distance
        # 这块要保证生成的机器人起始点和目标点之间的距离约束:
        # self.minimum_distance < 【distance】 < total_step / target_distance_gap + self.init_target_distance
        self.init_target_distance = init_target_distance  # 初始化机器人的起点和目标点的【最大】距离
        self.minimum_distance = 0  # 机器人起点和目标点的最小距离。
        self.record_history = record_history  # 是否记录历史数据
        self._start_pos = None  # 机器人起始点位置
        self.distance_to_barrier = None
        if self.record_history:
            self.history = History()  # 机器人的轨迹记录


    def save_history(self, file_path="/tmp/local_nav_history.pkl"):
        if self.record_history and hasattr(self, "history"):
            self.history.save(file_path)

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(-1, 1, shape=(self._obs_dim,))

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(3)  # F, L, R(前进、向左、向右)

    def eval_model(self, mini_distance:float = None, max_distance:float=None) -> None:
        # 模型评估阶段，每个episode的最大步数设置为正无穷。
        self.init_target_distance = np.inf
        if max_distance is not None:
            self.init_target_distance = max_distance
        if mini_distance is not None:
            self.minimum_distance = mini_distance
        self.episode_maximum_steps = 1024
        

    def _scan_callback(self, data: LaserScan) -> None:
        # 获得机器人的激光雷达数据
        self._scan_data = data

    def _odom_callback(self, data: Odometry) -> None:
        # 获得机器人的姿态数据
        self._odom_msg = data.pose.pose

    def _map_callback(self, data:OccupancyGrid) -> None:
        #获得地图数据
        self._map_data = data

    def _is_collision(self) -> bool:
        # 机器人碰撞检测

        # pos_x = self._odom_msg.position.x
        # pos_y = self._odom_msg.position.y
        return self.distance_to_barrier <= LocalConstants.CollisionDistance
        # return self.collision_checker.is_collision(pos_x, pos_y)

    def distance(self, position: np.ndarray) -> float:
        # 计算机器人当前位置和目标点的距离
        distance = np.linalg.norm(self.agent_target_pose[POSITION] - position)
        return distance

    def _seed(self, seed: int = None) -> List[float]:
        # 设置随机种子
        self.np_random, seed = seeding.np_random(seed=seed)
        return [seed]

    @staticmethod
    def discretize_laser_observation(data, new_ranges: float) -> np.ndarray:
        # 对激光雷达数据进行离散化处理, 并进行归一化
        discretized_ranges = []
        mod = len(data.ranges) / new_ranges
        for i, _ in enumerate(data.ranges):
            if i % mod == 0:
                if data.ranges[i] == float("Inf") or data.ranges[i] > 8:
                    discretized_ranges.append(0.0)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0.0)
                else:
                    discretized_ranges.append(data.ranges[i])
        laser_data = np.array(discretized_ranges)
        return laser_data / data.range_max  # 归一化到0-1之间

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_step += 1
        self.total_step += 1

        ############################################################################################################
        # 这个长注释块内让gazebo引擎运行，获得机器人各个状态信息；获得信息后引擎暂停，以便于下一步的处理。
        # pause the simulation engine
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/unpause_physics service call failed")

        # get sensor data.
        vel_cmd = self._scale_action_to_control(action)
        self.vel_pub.publish(vel_cmd)  # steps
        # get data.
        self.ros_sleep()  # 从ros通信中获取数据
        self.distance_to_barrier = np.min(np.array(self._scan_data.ranges))  #最小是0.45（碰撞）
        is_collision = self._is_collision()  # 碰撞检测
        position = np.array(
            [self._odom_msg.position.x, self._odom_msg.position.y]
        )  # 机器人当前位置
        # 朝向信息
        orientation = np.array(
            [
                self._odom_msg.orientation.w,
                self._odom_msg.orientation.x,
                self._odom_msg.orientation.y,
                self._odom_msg.orientation.z
                
            ]
        )
        laser_data = self._scan_data
        map_counter = Counter(self._map_data)
        explored_map_area = map_counter[100]+map_counter[0]
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/pause_physics service call failed")
        ############################################################################################################
        self.agent_current_pose[POSITION] = position
        self.agent_current_pose[ORIENTATION] = orientation
        laser_data = self.discretize_laser_observation(laser_data, 10)
        pos_state = self.get_pos_state(position, self.agent_target_pose[POSITION])
        # reward function
        state = np.concatenate((pos_state, orientation, laser_data))
        reward, done, status = self._get_reward(position, is_collision, action, explored_map_area)

        self._clear_data()
        info = {}
        # 添加记录信息
        if self.record_history:
            self.history.push(
                state=(position, orientation), action=action, reward=reward, done=done
            )

        if done:
            info[STATUS] = status
            if self.record_history:
                info["real_path_length"] = self.history[-2].traj_len  # -1为空的，-2才是最后一个
                info["minimum_path_length"] = np.linalg.norm(self._start_pos - self. agent_target_pose[POSITION])
                if status == FAILED:
                    info["terminal2target_length"] = np.linalg.norm(self.agent_current_pose[POSITION] - self.agent_target_pose[POSITION])
                else:
                    info["terminal2target_length"] = 0

        return state, reward, done, info

    def _get_reward(self, cur, is_collision, action, map) -> Tuple[float, bool, str, float]:
        reward = map / self.total_map_area
        reward *= 100
        if is_collision:
            return -100+reward, True, FAILED
        current_distance = self.distance(cur)
        # if success
        # if current_distance < LocalConstants.SuccessDistance:
        #     return 1000+reward, True, SUCCESS

        # 这里是为了让机器人保持直线行驶，而不是随意拐弯;
        if self.current_step >= self.episode_maximum_steps:
            # return -1000+reward., True, FAILED
            return reward, True, FAILED
        # reward = action[0] * np.cos(action[1]) * 0.5


        #  margin reward
        # if self.distance_to_barrier > 0.45 and self.distance_to_barrier < 0.53:
        #     reward_margin = -(1 / self.distance_to_barrier)
        # elif self.distance_to_barrier >=0.53 and self.distance_to_barrier <0.60:
        #     reward_margin = -(1-self.distance_to_barrier/0.60)
        # else:
        #     reward_margin = 0
        # reward += reward_margin


        return reward, False, RUNNING

    def reset(self, local_target_pose:np.ndarray=None) -> np.ndarray:
        """强化学习环境重置函数

        Args:
            local_target (np.ndarray, optional): 在作为导航局部规划时，目标点的位置. Defaults to None.
            是个六个纬度的向量，前三个是位置，后三个是四元数. local_target_pose = [x, y, z, qw, qx, qy, qz]

        Returns:
            _type_: state, np.ndarray
        """
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/unpause_physics service call failed")

        if not self._as_local_planner:
            self._reset_robot() # 重置机器人位置
        self.ros_sleep()
        position = np.array([self._odom_msg.position.x, self._odom_msg.position.y])
        orientation = np.array(
            [
                self._odom_msg.orientation.w,
                self._odom_msg.orientation.x,
                self._odom_msg.orientation.y,
                self._odom_msg.orientation.z,
                
            ]
        )
        laser_data = self._scan_data

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/pause_physics service call failed")

        # 添加记录
        if self.record_history:
            if not hasattr(self, "history"):  # 如果中途做测试需要记录，只需改变record_history的值即可
                self.history = History()
            self.history.push((position, orientation), np.zeros(2), 0, False)

        self._start_pos = position
        self.agent_current_pose[POSITION] = position
        self.agent_current_pose[ORIENTATION] = orientation
        
        if not self._as_local_planner:
            self.agent_target_pose[POSITION] = self.collision_checker.get_target(
                start_pos=position,
                max_distance=self.init_target_distance + self.total_step / self.target_distance_gap,
                min_distance=self.minimum_distance
            )
        else:
            self.agent_target_pose[POSITION] = np.array(local_target_pose[:2])
            self.agent_target_pose[ORIENTATION] = np.array(local_target_pose[3:])
            

        laser_data = self.discretize_laser_observation(laser_data, 10)  # laser_data
        pos_state = self.get_pos_state(position, self.agent_target_pose[POSITION])
        state = np.concatenate((pos_state, orientation, laser_data))
        self.current_step = 0
        self._clear_data()
        return state

    def _clear_data(self):
        # 这里清空数据的原因：需要保持每次step获得的传感器信息都是最新的
        self._odom_msg = None
        self._scan_data = None
        self._map_data = None

    def ros_sleep(self):
        # 这里的sleep是为了保证每次都能让对应的变量获得传感器信息
        while not rospy.core.is_shutdown() and (
                self._odom_msg is None or self._scan_data is None
        ):
            rospy.rostime.wallsleep(0.01)

    @staticmethod
    def get_pos_state(current_position:np.ndarray, target_position:np.ndarray) -> np.ndarray:
        # 机器人当前位置与目标位置的相对位置，返回值为一个二维向量，表示机器人当前位置与目标位置的距离和方向;
        distance = np.linalg.norm(current_position - target_position)
        angle = np.arctan2(
            target_position[1] - current_position[1],
            target_position[0] - current_position[0],
        ) / (2 * np.pi)  # 归一化
        return np.array([distance, angle])

    def _reset_robot(self):
        # 重置机器人位置
        cmd_vel = Twist()
        self.vel_pub.publish(cmd_vel)
        msg = ModelState()
        msg.model_name = "pioneer3at_robot"
        tmp = self.collision_checker.get_target()
        msg.pose.position.x = tmp[0]
        msg.pose.position.y = tmp[1]
        # msg.pose.position.x = 0
        # msg.pose.position.y = 0
        msg.pose.orientation.z = np.random.rand(1)[0]
        msg.pose.orientation.w = np.random.rand(1)[0]
        while True:
            try:
                self.set_gazebo(msg)
            except rospy.ServiceException as e:
                time.sleep(0.1)
            else:
                break

    def _scale_action_to_control(self, action) -> Twist:
        """
        # action: 网络输出的控制信息，这个函数将其转换为机器人的控制信息
        action = int, Discrete(3), [0, 1, 2]
        """
        vel_cmd = Twist()
        if action == 0:  # Forward
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 0.0
        elif action == 1:  # Left
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = 0.3
        elif action == 2:
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -0.3
        return vel_cmd

    def close(self):
        self._close()


class LocalNavContinusEnv(LocalNavEnv):
    """
    连续控制
    将接收的范围在[0, 1]的网络输出2维连续控制量（分别表示线速度和角速度），转换为控制机器人底层的线速度和角速度，角速度范围在[-0.3, 0.3]内；
    线速度在[0, 0.3]；
    """

    def __init__(
            self,
            train_config: Union[dict, str],
            episode_maximum_steps=512,
            init_target_distance=2,
            target_distance_gap=10000,
            record_history=False,
            as_local_planner=False

    ) -> None:
        """局部导航环境类【连续动作空间】，提供基本的env step、reset、render等方法
        Args:
            train_config (dict|str): 训练时的config字典，存储一些路径和常量信息
            episode_maximum_steps (int, optional):  每个episode的最大步数. Defaults to 512.
            init_target_distance (int, optional): 初始化机器人的起点和目标点的【最大】距离. Defaults to 2.
            target_distance_gap (int, optional): 表示每get_target_distance_gap步，目标点和出发点的距离会增加1. Defaults to 10000.
            record_history (bool, optional): 是否记录轨迹信息. Defaults to False.
            as_local_planner (bool, optional): 是否作为局部规划器. Defaults to False.
        """
        super().__init__(train_config, episode_maximum_steps, init_target_distance, target_distance_gap, record_history,as_local_planner)

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Box(-1, 1, (2,), dtype=np.float32)

    def _scale_action_to_control(self, action) -> Twist:
        """
        action = np.array([a, b])
        a: cmd_vel, range:(-1, 1) mapping to (0, 0.3)
        b: angle: range:(-1, 1), mapping to (-0.3, 0.3)
        """
        action[0] = (action[0] + 1) * 0.15  # cmd_vel
        action[1] = action[1] * 0.3
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]  # cmd_vel
        vel_cmd.angular.z = action[1]  # angle
        return vel_cmd


class LocalNavContinusIncrementalEnv(LocalNavContinusEnv):
    """
    incremental action space.
    增量式控制:
    $v_t = v_{t-1}+\delta v_t$
    """

    def __init__(
            self,
            train_config: Union[dict, str],
            episode_maximum_steps=512,
            init_target_distance=2,
            target_distance_gap=10000,
            record_history=False,
            as_local_planner=False
    ) -> None:
        """局部导航环境类【增量连续动作空间】，提供基本的env step、reset、render等方法
        Args:
            train_config (dict|str): 训练时的config字典，存储一些路径和常量信息
            episode_maximum_steps (int, optional):  每个episode的最大步数. Defaults to 512.
            init_target_distance (int, optional): 初始化机器人的起点和目标点的【最大】距离. Defaults to 2.
            target_distance_gap (int, optional): 表示每get_target_distance_gap步，目标点和出发点的距离会增加1. Defaults to 10000.
            record_history (bool, optional): 是否记录轨迹信息. Defaults to False.
            as_local_planner (bool, optional): 是否作为局部规划器. Defaults to False.
        """
        super().__init__(train_config, episode_maximum_steps, init_target_distance, target_distance_gap, record_history,as_local_planner)
        self.previous_velocity = np.array([0, 0])  # cmd_vel, angle
        # change observation space
        self._obs_dim = self._obs_dim + 2

    def _scale_action_to_control(self, action) -> Twist:
        """small change based on previous action"""
        action[0] = action[0] * 0.1 + self.previous_velocity[0]  # cmd_val
        action[1] = action[1] * 0.1 + self.previous_velocity[1]
        if action[0] > 0.3:
            action[0] = 0.3
        if action[0] <= 0.01:
            action[0] = 0.05
        # clip the range
        if action[1] > 0.3:
            action[1] = 0.3
        if action[1] < -0.3:
            action[1] = -0.3
        self.previous_velocity = np.array([action[0], action[1]])
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]  # cmd_vel
        vel_cmd.angular.z = action[1]
        return vel_cmd

    def reset(self, local_target_pose:np.ndarray=None) -> np.ndarray:
        self.previous_velocity = np.array([0, 0])
        state = super().reset(local_target_pose=local_target_pose)
        return np.concatenate((state, self.previous_velocity))

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, info = super().step(action)
        return np.concatenate((state, self.previous_velocity)), reward, done, info


class MultiframeLocalNavContinusEnv(LocalNavContinusIncrementalEnv):
    """
    桢迭代的连续控制环境，解决POMDP问题
    """

    def __init__(
            self,
            train_config: Union[dict, str],
            episode_maximum_steps=512,
            init_target_distance=2,
            target_distance_gap=10000,
            record_history=False,
            as_local_planner=False
    ) -> None:
        """局部导航环境类【增量连续动作空间，桢迭代，解决POMDP】，提供基本的env step、reset、render等方法
        Args:
            train_config (dict): 训练时的config字典，存储一些路径和常量信息
            episode_maximum_steps (int, optional):  每个episode的最大步数. Defaults to 512.
            init_target_distance (int, optional): 初始化机器人的起点和目标点的【最大】距离. Defaults to 2.
            target_distance_gap (int, optional): 表示每get_target_distance_gap步，目标点和出发点的距离会增加1. Defaults to 10000.
            record_history (bool, optional): 是否记录轨迹信息. Defaults to False.
        """
        super().__init__(train_config, episode_maximum_steps, init_target_distance, target_distance_gap, record_history,as_local_planner)
        self.stack_nums = 4
        self._obs_dim = self.stack_nums * self._obs_dim

    def reset(self, local_target_pose:np.ndarray=None) -> np.ndarray:
        self.stack_frames = []
        state = super().reset(local_target_pose)
        for i in range(self.stack_nums):
            self.stack_frames.append(state)
        return np.concatenate(self.stack_frames)

    def step(self, action):
        self.stack_frames.pop(0)
        state, reward, done, info = super().step(action)
        self.stack_frames.append(state)
        return np.concatenate(self.stack_frames), reward, done, info



class LBULEnv(LocalNavContinusEnv):

    def __init__(self, train_config: Union[dict, str], episode_maximum_steps=512, init_target_distance=2,
                 target_distance_gap=10000, record_history=False, action_repeat=4,as_local_planner=False) -> None:
        """局部导航环境类【连续动作空间】，提供基本的env step、reset、render等方法
        Args:
            train_config (dict): 训练时的config字典，存储一些路径和常量信息
            episode_maximum_steps (int, optional):  每个episode的最大步数. Defaults to 512.
            init_target_distance (int, optional): 初始化机器人的起点和目标点的【最大】距离. Defaults to 2.
            target_distance_gap (int, optional): 表示每get_target_distance_gap步，目标点和出发点的距离会增加1. Defaults to 10000.
            record_history (bool, optional): 是否记录轨迹信息. Defaults to False.
            action_repeat(int, optional): 表示每个动作重复执行的次数. Defaults to 4.
            as_local_planner(bool, optional): 表示是否作为局部规划器. Defaults to False.
        """
        super().__init__(train_config, episode_maximum_steps, init_target_distance, target_distance_gap, record_history,as_local_planner)
        self.action_repeat = action_repeat

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        previous_distance = self.distance(self.agent_current_pose[POSITION])
        for _ in range(self.action_repeat):
            state, __, done, info = super().step(action)
            if done:
                break
        current_distance = self.distance(self.agent_current_pose[POSITION])
        reward = previous_distance - current_distance
        return state, reward, done, info


class FixOrientationEnv(LocalNavContinusIncrementalEnv):
    def __init__(
            self,
            train_config: Union[dict, str],
            episode_maximum_steps=512,
            init_target_distance=2,
            target_distance_gap=10000,
            record_history=False,
            as_local_planner=False
    ) -> None:
        super(FixOrientationEnv, self).__init__(train_config, episode_maximum_steps, init_target_distance, target_distance_gap, record_history, as_local_planner)
        self._obs_dim = self._obs_dim + 2

    def reset(self, local_target_pose:np.ndarray=None) -> np.ndarray:
        state = super().reset(local_target_pose=local_target_pose)
        # generate orientation via quaternion, for the ground which satisfy [w, x, y, z], x=0, y=0
        if not self._as_local_planner:
            # 随机生成姿态
            target_orient_w = np.random.rand(1)[0]
            target_orient_z = np.random.rand(1)[0]
            current_target_orient = np.array([target_orient_w, 0, 0, target_orient_z])
            self.agent_target_pose[ORIENTATION] = current_target_orient / np.linalg.norm(current_target_orient)
        
        w, z = self.agent_target_pose[ORIENTATION][0], self.agent_target_pose[ORIENTATION][3]
        tmp = np.array([w, z])
        state = np.concatenate((tmp, state))
        return state

    def step(self, action):
        state, reward, done, info = super().step(action)
        w, z = self.agent_target_pose[ORIENTATION][0], self.agent_target_pose[ORIENTATION][3]
        tmp = np.array([w, z])
        state = np.concatenate((tmp, state))
        # target pose (orientation)

        if  done and info[STATUS]==SUCCESS:
            target_quat = np.array([w, 0, 0, z])  # w, x, y, z
            target_quat2euler = np.rad2deg(quaternion2euler(target_quat)[2]) % 360

            # current pose (orientation)
            orientation = self.agent_current_pose[ORIENTATION]  # w,x,y,z
            current_quat2euler = np.rad2deg(quaternion2euler(orientation)[2]) % 360
            angle_difference = abs(target_quat2euler - current_quat2euler)
            if angle_difference > 180:
                angle_difference = 360.0 - angle_difference
            orientation_reward = 1 - angle_difference / 180.0
            reward += orientation_reward * 100
        return state, reward, done, info


    

class FrameStackEnv(FixOrientationEnv):
    def __init__(
            self,
            train_config: Union[dict, str],
            episode_maximum_steps=512,
            init_target_distance=2,
            target_distance_gap=10000,
            record_history=False,
            as_local_planner=False,
            stack_frame=4,
    ) -> None:
        """局部导航环境类【连续动作空间】,多帧迭代解决pomdp的问题，提供基本的env step、reset、render等方法

        Args:
            train_config (Union[dict, str]): 训练时的config字典，存储一些路径和常量信息
            episode_maximum_steps (int, optional): 每个episode的最大步数. Defaults to 512.
            init_target_distance (int, optional): 初始化机器人的起点和目标点的【最大】距离. Defaults to 2.
            target_distance_gap (int, optional): 表示每get_target_distance_gap步，目标点和出发点的距离会增加1. Defaults to 10000.
            record_history (bool, optional): 是否记录轨迹信息. Defaults to False.
            as_local_planner(bool, optional): 表示是否作为局部规划器. Defaults to False.
            stack_frame (int, optional): 连续多少帧作为一个状态. Defaults to 4.
        """
        super(FrameStackEnv, self).__init__(train_config, episode_maximum_steps, init_target_distance, target_distance_gap, record_history,as_local_planner)
        self._obs_dim = self._obs_dim * stack_frame
        self.stack_frame = stack_frame
        self.frame_buffer = deque(maxlen=stack_frame)

    def reset(self, local_target_pose:np.ndarray=None) -> np.ndarray:
        state = super().reset(local_target_pose=local_target_pose)
        for _ in range(self.stack_frame):
            self.frame_buffer.append(state)
        return np.concatenate(self.frame_buffer)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.frame_buffer.append(state)
        return np.concatenate(self.frame_buffer), reward, done, info
    
class DreamerEnv(LocalNavContinusEnv):

    def __init__(self, train_config: Union[dict, str], episode_maximum_steps=256, init_target_distance=2, target_distance_gap=10000, record_history=False, as_local_planner=False) -> None:
        super().__init__(train_config, episode_maximum_steps, init_target_distance, target_distance_gap, record_history, as_local_planner)
        self._obs_dim += 2
        self._prev_distance = None
        self._global_done = False
        self._prev_state = None

    def reset(self, local_target_pose: np.ndarray = None) -> np.ndarray:
        self._global_done = False
        state = super().reset(local_target_pose)
        current_steps = self.current_step / self.episode_maximum_steps
        state = np.concatenate([np.array([0, current_steps]), state], axis=0)
        self._prev_distance = self.distance(self.agent_current_pose[POSITION])
        self._prev_state = state
        return state
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        if self._global_done:
            self.current_step += 1
            state = self._prev_state
            state[0] = 1.
            state[1] = self.current_step / self.episode_maximum_steps
            done = True if self.current_step >= self.episode_maximum_steps else False
            return state, 0, done , {}
        
        state, _, done, info = super().step(action)
        if done: 
            self._global_done = True
        distance  = self.distance(self.agent_current_pose[POSITION])
        reward = self._prev_distance - distance
        
        current_steps = self.current_step / self.episode_maximum_steps

        flag = 1. if self._global_done else 0
        state = np.concatenate([np.array([flag, current_steps]), state], axis=0)
        self._prev_state = state
        self._prev_distance = distance
        
        fake_done = False
        if self.current_step >= self.episode_maximum_steps:
            fake_done = True
        return state, reward, fake_done, info


    def _get_reward(self, cur, is_collision, action) -> Tuple[float, bool, str]:
        if is_collision:
            return -0.1, True, FAILED
        current_distance = self.distance(cur)
        # if success
        if current_distance < 0.5:
            return 1., True, SUCCESS

        # 这里是为了让机器人保持直线行驶，而不是随意拐弯;
        # reward += action[0] * np.cos(action[1]) * 0.5
        if self.current_step >= self.episode_maximum_steps:
            return -1., True, FAILED
        reward = -0.0005
        return reward, False, RUNNING        
        

if __name__ == "__main__":
    from rl.scripts.utils.tools import load_train_config
    train_config_path = "rl/configs/train_config.yaml"
    train_config = load_train_config(train_config_path)
    env = LocalNavContinusIncrementalEnv(train_config=train_config, record_history=True)
    rollout_nums = 10
    for _ in range(rollout_nums):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                break

    for idx, records in enumerate(env.history):
        print(f"episode:{idx}, trajectory len:{records.traj_len}")
