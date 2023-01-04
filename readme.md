

## 安装和使用

### ROS工具包安装
**步骤：**

* 1.参考官网安装ROS和Gazebo，安装参考链接：[http://wiki.ros.org/ROS/Installation](http://wiki.ros.org/ROS/Installation)
* 2.安装智能车需要的工具包
    * velodyne: `sudo apt-get install ros-你的版本名称-velodyne ros-你的版本名称-velodyne-description`, eg: `sudo apt-get install ros-noetic-velodyne`
* 3.创建工作空间
    * `mkdir -p ~/catkin_ws/src`
    * `cd ~/catkin_ws/src`
    * `catkin_init_workspace`
* 4.克隆仓库
    * cd ~/
    * `git clone https://gitee.com/mahongying/navigation.git`
    * 构建工具包软连接: `ln -s ~/navigation ~/catkin_ws/src`
* 5.编译工作空间
    * `cd ~/catkin_ws`
    * `catkin_make`
* 6.设置环境变量    
    * `echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc`
    * `echo "export PYTHONPATH=~/catkin_ws/src" >> ~/.bashrc` #将python包路径添加到环境变量中，方便调用；路径为从根目录到src文件夹下
    * `source ~/.bashrc`
* 7.测试
    * 测试gazebo环境是否安装成功
        * `roslaunch amr_robots_gazebo example-pioneer3at-ampt.launch`
    * 测试强化环境
        * `cd ~/catkin_ws/src/rl/scripts/script_test && pytest`
* 8.强化学习智能体训练
    * `cd ~/catkin_ws/src/`
    * `python rl/train/train_local_avoidance.py --config rl/configs/train_config.yaml --log_dir tmp/rl`


### 局部避障

已有环境使用:
```python
# 环境采用标准gym接口，可直接适配主流RL算法框架（stable baseline、RLlib等）
# import 方式最后阶段会改进
from rl.scripts.envs.env.local_env.local_nav_env import GazeboEnv
train_config_path = 'rl/configs/local_nav/train.yaml'
env = LocalNavEnv(train_config=train_config_path,recorder_history=True)

env.reset()
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action) 
    if done:
        break
env.save_history(file_path='./trajectory_data.pkl')
env.close()
```

自定义环境使用:
```python
# import 方式最后阶段会改进
from rl.scripts.envs.env.local_env.local_nav_env import GazeboEnv

class YourEnv(GazeboEnv):
    def __init__(self, train_config, **kwargs):
        super(YourEnv, self).__init__(train_config, **kwargs)
        # 重写初始化函数，定义自己的环境
    
    def reset(self):
        # 重写reset函数，定义自己的reset逻辑
        return obs
    
    def step(self, action):
        # 重写step函数，定义自己的step逻辑
        return obs, reward, done, info

#使用
env = YourEnv(train_config=train_config_path,**kwargs)
env.reset()
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action) 
    if done:
        break
env.save_history(file_path='./trajectory_data.pkl')
env.close()
```