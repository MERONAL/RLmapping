import sys
import os
import signal
import subprocess
import random
from typing import Union
import gym
import rospy

from rosgraph_msgs.msg import Clock


class GazeboEnv(gym.Env):
    '''
    super-class for all gazebo envs.
    '''
    metadata = {"render.modes" :['human']}

    def __init__(self, train_config: Union[dict, str]) -> None:
        """Gazebo环境的基类，所有的Gazebo环境都应该继承此类，需要实现step和reset方法

        Args:
            train_config (Union[dict, str]): 训练配置文件，可以是一个字典，也可以是一个文件路径

        Raises:
            IOError: 如果train_config是一个文件路径，但是文件不存在，会抛出IOError
        """
        if isinstance(train_config, str):
            from rl.scripts.utils.tools import load_train_config
            self.train_config = load_train_config(train_config)
        else:
            self.train_config = train_config
        launch_filepath = self.train_config['gazebo']['launch_file_path']
        world_filepath = self.train_config['gazebo']['world_file_path']
        print("world file path:", world_filepath)
        self.last_clock_msg = Clock()
        random_number = random.randint(10000, 15000)
        self.port = str(random_number)
        self.port_gazebo = str(random_number + 1)
        os.environ["ROS_MASTER_URI"] = "http://localhost:" + self.port
        os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + self.port_gazebo
        print(f"ROS_MASTER_URI=http://localhost:{self.port}\n")
        print(f"GAZEBO_MASTER_URI=http://localhost:{self.port_gazebo}\n")
        # ros_path: opt/ros/neotic/bin       
        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))

        if launch_filepath.startswith('/'):
            fullpath = launch_filepath
        else:
            fullpath = os.path.join(os.path.dirname(__file__),'assets',"launch", launch_filepath)
        
        if not os.path.exists(fullpath):
            raise IOError("File " + fullpath+ " does not exist." )
        
        self._roslaunch = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", self.port, fullpath, f"world:={world_filepath}"])
        print("Gazebo launched!")
        self.gzclient_pid = 0

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
    
    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
    
    def _render(self, mode='human', close=False):
        if close:
            tmp = os.popen('ps -Af').read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen('ps -Af').read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(subprocess.check_output(["pidof", "-s", "gzclient"]))
        else:
            self.gzclient_pid = 0
    
    def _close(self):
        #kill gzclient, gzserver, and roscore
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count("gzserver")
        roscore_count = tmp.count("roscore")
        rosmaster_count = tmp.count("rosmaster")

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")
        
        if(gzclient_count or gzserver_count or roscore_count or rosmaster_count > 0):
            os.wait()
    
    def _seed(self):
        #set random seed for current envs.
        pass

if __name__ == '__main__':
    from rl.scripts.utils.tools import load_train_config
    train_config_path = 'rl/configs/train_config.yaml'
    train_config = load_train_config(train_config_path)
    env = GazeboEnv(train_config=train_config)
