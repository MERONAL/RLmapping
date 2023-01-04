from typing import List, Tuple
import numpy as np


class Pose:
    def __init__(self, position, orientation) -> None:
        self.position = position
        self.orientation = orientation
    
def euclidean_distance(pos1: Pose, pos2: Pose) -> float:
    """Euclidean distance between two poses."""
    return np.sqrt((pos1.position[0] - pos2.position[0])**2 + (pos1.position[1] - pos2.position[1])**2)

class Trajectory:
    '''
    记录一个episode的信息，包括：状态、动作、和累积奖励;
    '''
    def __init__(self):
        self.states :List[Pose] = list()
        self.actions = list()
        self.episode_rewards = 0

    def push(self, state:Pose, action:np.ndarray, reward:float) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.episode_rewards += reward
    
    @property
    def traj_len(self) -> float:
        '''
        获得轨迹实际的物理距离
        '''
        traj_len = 0
        for i in range(len(self.states) - 1):
            traj_len += euclidean_distance(self.states[i], self.states[i+1])
        return traj_len
        
    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index]

    def __iter__(self):
        return iter(zip(self.states, self.actions, self.rewards))

class History:
    '''
    记录智能体的轨迹历史记录。History主要由一系列的Trajectory组成，每个Trajectory记录一个episode的信息，包括：状态、动作、和累积奖励;
    '''
    def __init__(self) -> None:
        self.traj :List[Trajectory] = list() 
    
    def __len__(self) -> int:
        return len(self.traj)
    
    def push(self, state:Tuple[np.ndarray, np.ndarray], action:np.ndarray, reward:float, done:bool) -> None:
        if len(self.traj) == 0:
            self.traj.append(Trajectory())
        
        pose = Pose(position=state[0], orientation=state[1])
        self.traj[-1].push(pose, action, reward)
        if done:
            self.traj.append(Trajectory())
            
    def __getitem__(self, index):
        return self.traj[index]

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.traj, f)
    
    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.traj = pickle.load(f)
       


def euler2quaternion(roll:float, pitch:float, yaw:float) -> List[float]:
    """将欧拉角转换为四元数,弧度单位
    Args:
        roll (float): 表示绕x轴的旋转角度,横滚角
        pitch (float): 表示绕y轴旋转的角度, 俯仰角
        yaw (float): 表示绕z轴旋转的角度,偏航角

    Returns:
        List[float]: 四元数[w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return np.array([w, x, y, z])

def quaternion2euler(quaternion:list) -> List[float]:
    """四元数转换为欧拉角
    Args:
        w (float): 四元数w
        x (float): 四元数x
        y (float): 四元数y
        z (float): 四元数z
    Returns:
        List[float]: 欧拉角[roll, pitch, yaw]
    """
    unit_quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = unit_quaternion
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([roll, pitch, yaw])


