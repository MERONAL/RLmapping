import subprocess, sys, os
import math
import numpy as np
from rl.scripts.utils.tools import load_yaml


def load_scene_config(config:dict, scene:str):
    obstacle_radius = config['obstacle_info']['obstacle_radius']
    robot_radius = config['robot']['robot_size']
    collision_threshold = config['hyper_parameter']['collision_threshold']
    collision_distance = obstacle_radius + robot_radius + collision_threshold
    scene_info = config['scene'][scene]
    obstacles = scene_info['static_obstacle']
    walls = scene_info['walls']
    dynamic_obstacle_config = None
    if scene_info['dynamic']:
        dynamic_obstacle_config = scene_info['dynamic_obstacles']

    return obstacles, walls, collision_distance, robot_radius, dynamic_obstacle_config

class CollisionChecker:
    def __init__(self, scene_config:dict, scene:str):
        """碰撞检测和短途目标生成模块
        Args:
            scene_config (dict): 场景配置文件
                key
            scene (str): 场景名称
        """
        self.obstacles, self.walls, self.collision_distance,self.robot_radius, dynamic_obstacle_config = load_scene_config(scene_config, scene)
        self.dynamic_obstacle_data = None
        if dynamic_obstacle_config is not None:
            '''加载动态避障的进程'''
            import rospy
            from std_msgs.msg import Float32MultiArray
            python_path_dir = os.path.dirname(subprocess.check_output(["which", "python"])).decode('utf-8')   #存放python的目录文件夹
            python_path = os.path.join(python_path_dir, 'python3')      #python3的绝对路径
            ros_package_path = os.environ['ROS_PACKAGE_PATH']
            python_script_path =os.path.join(ros_package_path.split(':')[0],"gazebo/src/ros_dynamic_obstacle.py")
            assert os.path.isfile(python_script_path), f"`{python_script_path}` file not found"
            subprocess.Popen([python_path, python_script_path,  str(dynamic_obstacle_config)])
            self.obstacles_pose_subscriber = rospy.Subscriber('/dynamic_obstacle_pose',Float32MultiArray, callback=self._obstacle_cb)
    
    def _obstacle_cb(self, msg):
        self.dynamic_obstacle_data = np.array(msg.data).reshape(-1, 2).tolist()
        
    def is_collision(self, pos_x, pos_y) -> bool: 
        """碰撞检测
        Args:
            pos_x (float): 机器人的x坐标
            pos_y (float): 机器人的y坐标
            dynamic_obstacle_data (List[List], optional): 传入动态障碍物（圆柱体）的实时坐标，用于统计动态障碍物. Defaults to None.
        Returns:
            bool: 是否发生了碰撞 
        """
        if pos_x < self.walls[0] + self.robot_radius or pos_x > self.walls[1] - self.robot_radius or pos_y > self.walls[2] - self.robot_radius or pos_y < self.walls[3] + self.robot_radius: return True
        if self.dynamic_obstacle_data is None: # for select the target.
            object_data = self.obstacles
        else:
            object_data = self.dynamic_obstacle_data + self.obstacles
        
        for item in object_data:
            dis = math.sqrt((pos_x - item[0]) **2 + (pos_y - item[1])**2)
            if dis < self.collision_distance: return True
        return False


    def get_target(self,start_pos:np.ndarray=np.array([0, 0]), max_distance=1000, min_distance = 0) -> np.ndarray:
        """生成随机的目标点；
        生成规则：
            根据当前位置和最大最小距离约束，生成距离机器人一定范围内的随机目标点；其中生成的目标点不能与障碍物发生碰撞，且满足:
            1. 机器人的位置到目标点的距离在[min_distance, max_distance]之间；
        Args:
            start_pos (np.ndarray, optional): 当前机器人的位置坐标np.array([x, y]). Defaults to np.array([0, 0]).
            max_distance (int, optional): 生成目标点距离机器人的最远距离约束. Defaults to Inf.
            min_distance (int, optional): 生成目标点距离机器人的最小距离约束. Defaults to None.
            dynamic_obstacle_data (List[List], optional):传入动态障碍物（圆柱体）的实时坐标，用于统计动态障碍物. Defaults to None.

        Returns:
            np.ndarray: 生成的目标点位置
        """        
        
        max_distance_square = max_distance * max_distance
        min_distance_square = min_distance * min_distance
        while True:
            pos_x = np.random.randint(self.walls[0] + 1, self.walls[1] - 1) + np.random.random()
            pos_y = np.random.randint(self.walls[3] + 1, self.walls[2] - 1) + np.random.random()
            dis = (start_pos[0] - pos_x) ** 2 + (start_pos[1] - pos_y) ** 2
            if dis > max_distance_square or dis <= min_distance_square:
                continue
            if not self.is_collision(pos_x, pos_y):
                break
        return np.array([pos_x, pos_y])
    
