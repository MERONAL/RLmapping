import os
import random
from copy import deepcopy
from collections import defaultdict
from queue import PriorityQueue
from typing import Union, List, Any, Tuple,  Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry, Path

from gazebo.src.new_map import TopologicalMap
from rl.scripts.envs.env.local_env.local_nav_env import LocalNavEnv, LocalNavContinusIncrementalEnv,LocalNavContinusEnv,FrameStackEnv
from rl.scripts.envs.utils import euler2quaternion
from rl.scripts.envs.keys import SUCCESS, FAILED,RUNNING, STATUS, LocalConstants, REAL_PATH, PLANNED_PATH


class GlobalPlanner:
    def __init__(self, map_file_path:str=None, ifplot :bool = False) -> None:
        """全局路径规划算法,给定地图和输入,返回一条路径.

        Args:
            map_file_path (str): 地图路径
        """
        self.map = TopologicalMap(ifplot=ifplot)
        if os.path.exists(map_file_path):
            self.map.load(map_file_path)
        else:
            raise FileNotFoundError(f"map file not found: {map_file_path}")
        self._tmp_path = None
        print(f"\033[1;32m[GlobalNavigation][INFO] successfully initialize GlobalPlanner \033[0m")
    
    def __call__(self,start_pos:Union[List[float], PoseStamped], target_pos:Union[List[float], PoseStamped]) -> Path:
        """全局路径规划算法,给定终点坐标或者姿态,返回一条路径.

        Args:
            start_pos (Union[List[float], PoseStamped]): 起点坐标或者姿态
            target_pos (Union[List[float], PoseStamped]): 目标点坐标或者姿态

        Returns:
            Path (nav_msgs.msg.Path): 路径
        """
        if isinstance(target_pos, PoseStamped):
            target_pos = [target_pos.pose.position.x, target_pos.pose.position.y]
        if isinstance(start_pos, PoseStamped):
            start_pos = [start_pos.pose.position.x, start_pos.pose.position.y]
        assert isinstance(target_pos, list) and isinstance(start_pos, list) and len(target_pos) == 2 and len(start_pos) == 2, "target_pos or start_pos is not valid"
        pathes = self.map.path_planning(start_pos, target_pos)
        self._tmp_path = pathes
        pathes_ros = self.map.path2ros_path(pathes)
        return pathes_ros
    
    def plot(self, show_path=True):
        if show_path:
            self.map.plot(path=self._tmp_path)
        else:
            self.map.plot()
    
    def update_node(self, pose: PoseStamped) -> None:
        """更新地图节点信息.

        Args:
            pose (PoseStamped): 当前位置
        """
        x, y = pose.pose.position.x, pose.pose.position.y
        self.map.new_node(x, y)
    
class LocalPlanner:
    def __init__(
            self,
            env:LocalNavEnv,
            model_config:dict,
    ) -> None:
        """局部避障算法，给定环境和输入，完成一次局部规划任务

        Args:
            env (dict): 环境
            model_config: 模型配置
        """

        self.local_env :LocalNavEnv = env
        assert isinstance(self.local_env, LocalNavEnv), "env is not a subclass of LocalNavEnv"
        # 保证模型可以是实现：action = self.local_model(state)
        self.action_model = self.load_action_model(**model_config)
        assert callable(self.action_model), "action_model is not callable"
        self._long_episode_info = {}
        print(f"\033[1;32m[GlobalNavigation][INFO] successfully initialize LocalPlanner \033[0m")

    
    
    @staticmethod
    def load_action_model(*args, **kwargs) -> callable:
        """加载决策模型"""
        # 写个stable baselines的加载模型的函数作为例子，可以是任何的决策模型
        from stable_baselines3 import SAC
        model_path = kwargs.get("model_path", None)
        print(kwargs)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model file not found: {model_path}")
        model = SAC.load(model_path)
        return model.predict
    
    
    
    def reset(self, local_target_pose:Union[List[float], PoseStamped]) -> Any:
        """重置局部环境
        Args:
            local_target_pose (Union[List[float], PoseStamped]): 局部目标点坐标或者姿态
        """
        if isinstance(local_target_pose, PoseStamped):
            # x, y, z, qw, qx, qy, qz
            local_target_pose = [local_target_pose.pose.position.x, local_target_pose.pose.position.y, 0, 1,0,0,0]
        assert isinstance(local_target_pose, list) and len(local_target_pose) == 7, "local_target_pose is not valid"
        self._long_episode_info = {}
        self._long_episode_info['reward'] = 0
        state = self.local_env.reset(local_target_pose=local_target_pose)
        return state
    
    def step(self, state) -> Union[Any, bool]:
        action, _ = self.action_model(state)
        state, reward, done, info = self.local_env.step(action=action)
        self._long_episode_info['reward'] += reward
        if done:
            self._long_episode_info['minimum_path_length'] = info['minimum_path_length']
            self._long_episode_info['real_path_length'] = info['real_path_length']
            if info[STATUS]==FAILED: print(f"\033[1;31m[LocalPlanner][ERROR] Collision was detected! \033[0m")

        return state, done, info
        
    @property
    def local_info(self) -> dict:
        return self._long_episode_info
  
def distance_between_two_pose(p1:PoseStamped, p2:PoseStamped):
    """返回平面上两个姿态的欧式距离:

    Args:
        p1 (PoseStamped): 姿态1
        p2 (PoseStamped): 姿态2

    Returns:
        float: 距离
    """
    return np.sqrt((p1.pose.position.x - p2.pose.position.x)**2 + (p1.pose.position.y - p2.pose.position.y)**2)

def pose_from_A2B(point_A:PoseStamped, point_B: PoseStamped) -> PoseStamped:
    """计算从A点到B的朝向信息,用于局部目标点的设置

    Args:
        point_A (PoseStamped): A点
        point_B (PoseStamped): B点

    Returns:
        PoseStamped: 从A到B的位姿信息
    """
    ax, ay = point_A.pose.position.x, point_A.pose.position.y
    bx, by = point_B.pose.position.x, point_B.pose.position.y
    yaw = np.arctan2(by-ay, bx-ax)
    quaternion = euler2quaternion(0, 0, yaw)
    pose_stamp = PoseStamped()
    pose_stamp.pose.position.x = bx - ax
    pose_stamp.pose.position.y = by - ay
    pose_stamp.pose.position.z = 0
    pose_stamp.pose.orientation.w = quaternion[0]
    pose_stamp.pose.orientation.x = quaternion[1]
    pose_stamp.pose.orientation.y = quaternion[2]
    pose_stamp.pose.orientation.z = quaternion[3]
    return pose_stamp

def hash_posestamped(data:PoseStamped) -> int:
    x = data.pose.position.x
    y = data.pose.position.y
    return hash(f"{format(x, '.3f')}_{y, '.3f'}")


def hash_distance_key(data:PoseStamped, keys:list, radius: float=0.5) -> float:
    tmp_x = data.pose.position.x
    tmp_y = data.pose.position.y
    for (x, y) in keys:
        if np.sqrt((tmp_x - x)**2 + (tmp_y - y)**2) < radius:
            return (x, y)
    return (tmp_x, tmp_y)

def get_path_length(path:Path) -> float:
    """计算路径的长度

    Args:
        path (Path): 路径

    Returns:
        float: 路径长度
    """
    trajectory_length = 0
    for i in range(1, len(path.poses)):
        trajectory_length += distance_between_two_pose(path.poses[i-1], path.poses[i])
    return trajectory_length
    

class PoseStampedStruct:
    def __init__(self, pose:PoseStamped, planning_distance:int, real_distance:float) -> None:
        self.pose = pose
        self.planning_distance = planning_distance
        self.real_distance = real_distance
    
    def __lt__(self, other:'PoseStampedStruct') -> bool:
        return self.planning_distance / self.real_distance < other.planning_distance / other.real_distance




class GlobalNavigation:
    """该类主要用于全局
    """
    def __init__(self, env:LocalNavEnv, global_planner:GlobalPlanner, local_planner:LocalPlanner) -> None:
        assert isinstance(env, LocalNavEnv), "env is not a subclass of LocalNavEnv"
        assert isinstance(global_planner, GlobalPlanner), "global_planner is not a instance of GlobalPlanner"
        assert isinstance(local_planner, LocalPlanner), "local_planner is not a instance of LocalPlanner"
        self.env = env
        self.global_planner = global_planner
        self.local_planner = local_planner
        self.path_planning_pub = rospy.Publisher("/global_navigation_path", Path, queue_size=1)
        self.real_path_pub = rospy.Publisher("/navigation_real_path", Path, queue_size=1)
        self.local_target_pub = rospy.Publisher("/local_target_pose", PoseStamped, queue_size=10)
        self.next_local_target_pub = rospy.Publisher("/next_local_target", PointStamped, queue_size=1)
        self.real_path :Path = None
        self.global_info:Dict[str] = None
        self._current_pose :PoseStamped= None
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._global_planner_callback)
        rospy.Subscriber("/sim_p3at/odom", Odometry, self._odom_callback, queue_size=1)
        print(f"\033[1;32m[GlobalNavigation][INFO] successfully initialize Global Navigation System! \033[0m")

    
    def _odom_callback(self, odom_msg: Odometry) -> None:
        self._current_pose = PoseStamped()
        self._current_pose.pose = odom_msg.pose.pose
    
    @property
    def current_pose(self) -> PoseStamped:
        return self._current_pose

    def _global_planner_callback(self, global_target:PoseStamped,update_node:bool = True) -> None:
        """全局路径规划回调函数"""
        if self.current_pose is None: return 
        global_path:Path = self.global_planner(self.current_pose, global_target)
        re_pub_path = deepcopy(global_path)
        self.path_planning_pub.publish(global_path)
        local_target, next_local_target_point = self.local_target_selection(global_path, self.current_pose)
        if local_target is None: return
        real_path = Path()
        real_path.header.frame_id = "map"
        real_path.header.stamp = rospy.Time.now()
        real_path.poses = []
        if isinstance(local_target, int): local_target = global_target
        
        self.local_target_pub.publish(local_target)
        if isinstance(next_local_target_point, PointStamped):
            self.next_local_target_pub.publish(next_local_target_point)
        state = self.local_planner.reset(local_target)
        real_path.poses.append(self.current_pose)
        tmp_steps = defaultdict(int)
        visited = set()
        visited.add(hash_posestamped(local_target))
        global_info = {}
        while distance_between_two_pose(self.current_pose, global_target) > LocalConstants.SuccessDistance:
            hashed_key = hash_distance_key(self.current_pose, list(tmp_steps.keys()), radius=1)
            tmp_steps[hashed_key] += 1
            state, done, local_info = self.local_planner.step(state)
            if update_node: self.global_planner.update_node(self.current_pose) # 更新拓扑地图
            real_path.poses.append(self.current_pose)
            local_distance = distance_between_two_pose(self.current_pose, local_target)
            if local_target != global_target and local_distance < 1:
                done = True
                local_info[STATUS] = SUCCESS
            
            if tmp_steps[hashed_key] > 512:
                done = True
                local_info[STATUS] = RUNNING
            
            if done:
                if local_info[STATUS] == FAILED:
                    global_info[STATUS] = FAILED
                    break
                next_local_target_point = None
                distance = distance_between_two_pose(self.current_pose, global_target)
                if distance < LocalConstants.SuccessDistance:
                    global_info[STATUS] = SUCCESS
                    break
                elif distance < 1:
                    local_target = global_target 
                    next_local_target_point = PointStamped()
                    next_local_target_point.header.frame_id = 'map'
                    next_local_target_point.header.stamp = rospy.Time.now()
                    next_local_target_point.point.x = global_target.pose.position.x
                    next_local_target_point.point.y = global_target.pose.position.y   
                else:             
                    path:Path = self.global_planner(self.current_pose, global_target)
                    self.path_planning_pub.publish(path)
                    if_choose_close_point = (local_info[STATUS] == RUNNING)
                    if if_choose_close_point:
                        tmp_steps = defaultdict(int)
                    prev_locat_target = local_target
                    local_target, next_local_target_point = self.local_target_selection(path, self.current_pose, visited, choose_closest_point=if_choose_close_point)
                    if local_target is None: 
                        local_target = global_target 
                        next_local_target_point = None
                        if local_target == prev_locat_target: 
                            global_info[STATUS] = FAILED
                            break
                    if isinstance(local_target, int): local_target = global_target
                    visited.add(hash_posestamped(local_target))
                state = self.local_planner.reset(local_target)
                self.local_target_pub.publish(local_target)
                if isinstance(next_local_target_point, PointStamped):
                    self.next_local_target_pub.publish(next_local_target_point)
                

        self.real_path_pub.publish(real_path)
        re_pub_path.header.stamp = rospy.Time.now()
        self.path_planning_pub.publish(re_pub_path)
        global_info[REAL_PATH] = real_path
        global_info[PLANNED_PATH] = re_pub_path
        if STATUS not in global_info:
            global_info[STATUS] = SUCCESS
        self.global_info = global_info

    @staticmethod
    def local_target_selection(global_path:Path, current_pose:PoseStamped, visited=set(),need_orientation=True, choose_closest_point: bool=False) -> Tuple[Union[PoseStamped, int], PointStamped]:
        """局部目标点选择
        Args:
            global_path (Path): 全局规划出来的路径信息
            current_pose (PoseStamped): 起始点的位姿态
            orientation (bool, optional): _description_. Defaults to False.

        Returns:
            Union[PoseStamped, int]: _description_
        """
        return GlobalNavigation.local_target_selection_origin(global_path, current_pose, visited, need_orientation,choose_closest_point)
        

    @staticmethod
    def local_target_selection_origin(global_path:Path, current_pose:PoseStamped, visited=set(),need_orientation=True,choose_closest_point: bool=False) -> Tuple[Union[PoseStamped, int], PointStamped]:
        """局部目标点选择
        Args:
            global_path (Path): 全局规划出来的路径信息
            current_pose (PoseStamped): 起始点的位姿态
            orientation (bool, optional): _description_. Defaults to False.

        Returns:
            Union[PoseStamped, int]: _description_
        """
        
        if global_path is None: return None, None
        if len(global_path.poses) == 0: return 0, None
        
        result_pose = PoseStamped()
        result_pose.header.frame_id = 'map'
        result_pose.header.stamp = rospy.Time.now()
        for idx, pose in enumerate(global_path.poses):
            if hash_posestamped(pose) in visited: 
                continue
            result_pose.pose = pose.pose
            if not need_orientation: return result_pose, None
            else:
                global_path.poses = global_path.poses[idx:]
                next_pose, _ = GlobalNavigation.local_target_selection(global_path, pose, need_orientation=False)
                next_local_target_point = None
                if isinstance(next_pose, PoseStamped):
                    result_pose.pose.orientation = pose_from_A2B(pose, next_pose).pose.orientation
                    next_local_target_point = PointStamped()
                    next_local_target_point.header.frame_id = 'map'
                    next_local_target_point.header.stamp = rospy.Time.now()
                    next_local_target_point.point.x = next_pose.pose.position.x
                    next_local_target_point.point.y = next_pose.pose.position.y
                
                return result_pose, next_local_target_point
        return None, None

    @staticmethod
    def local_target_selection_priority(global_path:Path, current_pose:PoseStamped, visited=set(),need_orientation=True,choose_closest_point: bool=False) -> Tuple[Union[PoseStamped, int], PointStamped]:
        """局部目标点选择
        Args:
            global_path (Path): 全局规划出来的路径信息
            current_pose (PoseStamped): 起始点的位姿态
            orientation (bool, optional): _description_. Defaults to False.

        Returns:
            Union[PoseStamped, int]: _description_
        """
        if global_path is None: return None, None
        if len(global_path.poses) == 0: return 0, None
        pose_queue = PriorityQueue()
        for idx, pose in enumerate(global_path.poses):
            distance = distance_between_two_pose(current_pose, pose)
            if distance < 1.5 or hash_posestamped(pose) in visited: continue
            tmp = PoseStampedStruct(pose, idx, -distance)
            pose_queue.put(tmp)
        if pose_queue.empty(): return 0, None
        pose = pose_queue.get()
        res_pose = pose.pose
        res_pose.header.frame_id = 'map'
        res_pose.header.stamp = rospy.Time.now()
        if not need_orientation: return res_pose, None
        idx = pose.planning_distance
        global_path.poses = global_path.poses[idx:]
        next_pose, _ = GlobalNavigation.local_target_selection(global_path, res_pose, visited, False)
        if isinstance(next_pose, PoseStamped):
            res_pose.pose.orientation = pose_from_A2B(res_pose, next_pose).pose.orientation
            next_local_target_point = PointStamped()
            next_local_target_point.header.frame_id = 'map'
            next_local_target_point.header.stamp = rospy.Time.now()
            next_local_target_point.point.x = next_pose.pose.position.x
            next_local_target_point.point.y = next_pose.pose.position.y
            return res_pose, next_local_target_point
        return res_pose, None


            
    @staticmethod
    def local_target_selection_graph_constraint(global_path:Path, current_pose:PoseStamped, visited=set(),need_orientation=True,choose_closest_point: bool=False) -> Tuple[Union[PoseStamped, int], PointStamped]:
        """局部目标点选择
        Args:
            global_path (Path): 全局规划出来的路径信息
            current_pose (PoseStamped): 起始点的位姿态
            orientation (bool, optional): _description_. Defaults to False.

        Returns:
            Union[PoseStamped, int]: _description_
        """
        if global_path is None: return None, None
        if len(global_path.poses) == 0: return 0, None
        def get_param_for_expontial_distribution(poses:List[PoseStamped]) -> float:
            """求解参数eta1,eta2和配分函数值
            Args:
                poses (_type_): path中PoseTamped构成的列表
            """
            traj_length = [0] * len(poses)
            distance_from_start = [0] * len(poses)
            for i in range(1, len(poses)):
                traj_length[i] = traj_length[i - 1] + distance_between_two_pose(poses[i], poses[i-1])
                distance_from_start[i] = distance_between_two_pose(poses[0], poses[i])
            eta2 = 1.
            eta1 = 1e-10
            for i in range(1, len(poses)):
                for j in range(i + 1, len(poses)):
                    delta_y = traj_length[j] - traj_length[i]
                    delta_x = distance_from_start[j] - distance_from_start[i]
                    val = delta_y / delta_x if abs(delta_x) > 1e-10 else 10
                    eta1 = max(eta1, val)
            logits = []
            for i in range(1, len(poses)):
                value = eta2 * traj_length[i]  - eta1 * distance_from_start[i]
                logits.append(value)
            # calculate the softmax
            logits = np.array(logits)
            logits_exp = np.exp(logits - logits.max())
            return logits_exp / logits_exp.sum()
            
        poses = [current_pose]
        idxes = [0]
        current_distance = 0
        distance_threshold = 3 if not choose_closest_point else 1
        for idx in range(1, len(global_path.poses)):
            current_distance += distance_between_two_pose(global_path.poses[idx], global_path.poses[idx-1])
            if current_distance < distance_threshold or hash_posestamped(global_path.poses[idx]) in visited: continue
            poses.append(global_path.poses[idx])
            idxes.append(idx)
            
            
        
        if len(poses) == 1: return 0, None
        prob = get_param_for_expontial_distribution(poses=poses)
        poses.pop(0)
        idxes.pop(0)
        # index = np.random.choice(range(len(poses)), p=prob)
        index = np.argmax(prob)
        res_pose = poses[index]
        idx = idxes[index]
        res_pose.header.frame_id = 'map'
        res_pose.header.stamp = rospy.Time.now()
        if not need_orientation: return res_pose, None
        global_path.poses = global_path.poses[idx:]
        next_pose, _ = GlobalNavigation.local_target_selection(global_path, res_pose, visited, False)
        if isinstance(next_pose, PoseStamped):
            res_pose.pose.orientation = pose_from_A2B(res_pose, next_pose).pose.orientation
            next_local_target_point = PointStamped()
            next_local_target_point.header.frame_id = 'map'
            next_local_target_point.header.stamp = rospy.Time.now()
            next_local_target_point.point.x = next_pose.pose.position.x
            next_local_target_point.point.y = next_pose.pose.position.y
            return res_pose, next_local_target_point
        return res_pose, None
    
    def local_test(self, episodes=50):
        """测试算法的一些指标:包括SPL,SR, NDG
        Args:
            episode (int, optional): 测试的轮数, Defaults to 50.
        """
        print("=====================================","LOCAL TEST","=====================================")
        collision_model  = self.local_planner.local_env.collision_checker
        spl = 0
        success_times = 0
        ndgs = np.zeros((episodes, ))
        for idx in range(episodes):
            x, y = collision_model.get_target()
            global_target = PoseStamped()
            global_target.header.frame_id = 'map'
            global_target.header.stamp = rospy.Time.now()
            global_target.pose.position.x = x
            global_target.pose.position.y = y
            self.global_info = None
            minmum_path_length = distance_between_two_pose(self.current_pose, global_target)
            self._global_planner_callback(global_target=global_target,update_node=False)
            while self.global_info is None:
                rospy.sleep(0.01)
            terminal2target = distance_between_two_pose(self.current_pose, global_target)
            ndgs[idx] = terminal2target / minmum_path_length
            try:
                if self.global_info[STATUS] == SUCCESS:
                    spl += minmum_path_length / max(minmum_path_length, get_path_length(self.global_info[REAL_PATH]))
                    success_times += 1
                    ndgs[idx] = 0
                elif self.global_info[STATUS] == FAILED:
                    self.local_planner.local_env._reset_robot()
                    self.local_planner.local_env.ros_sleep()
            except:
                print(self.global_info)
                break
            print("episode: {}, spl: {}, status: {}, ndg: {}".format(idx, spl/(idx + 1), self.global_info[STATUS], ndgs[idx]))
        print(f"\033[1;32m[GlobalNavigation][TEST INFO] SR:{success_times/episodes}, SPL:{spl/episodes}, NDG:{ndgs.mean()} \033[0m")
            
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="global navigation")
    parser.add_argument("--env_config", type=str, default="rl/configs/train_config.yaml", help="环境的配置文件")
    parser.add_argument("--map_file_path", type=str,default="rl/configs/maps/simple5_topologicalmap.pkl", help="地图文件路径")
    parser.add_argument("--model_path", type=str,default="/home/mhy/exp/10-19/SAC_localconNav.zip",help="局部路径规划模型路径")
    args = parser.parse_args()
    env = FrameStackEnv(args.env_config, record_history=True, as_local_planner=True)
    model_config = {"model_path": args.model_path}
    local_planner = LocalPlanner(env, model_config)
    global_planner = GlobalPlanner(args.map_file_path, ifplot=False)
    navigation = GlobalNavigation(env, global_planner, local_planner)
    rospy.sleep(5)
    navigation.local_test(episodes=50)
    rospy.spin()

    
        
        