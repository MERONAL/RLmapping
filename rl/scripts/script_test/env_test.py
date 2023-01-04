
def test_env_global_path():
    import os
    assert 'ROS_PACKAGE_PATH' in os.environ and os.environ['ROS_PACKAGE_PATH'] != ''

def test_scripts_utils():
    '''Test the scripts/utils.py module'''
    import os
    import numpy as np
    from rl.scripts.envs.utils import History
    history = History()
    for i in range(20):
        history.push((np.array([i, i]), np.array([i, i])), np.array([i, i]), i, False if i!=10 else True)
    history.push((np.array([20, 20]), np.array([20, 20])), np.array([20, 20]), 20, True)
    assert len(history) == 3
    assert len(history[0]) == 11
    assert abs(history[0].traj_len - 10 * np.sqrt(2)) < 1e-6
    history.save('/tmp/test.pkl')
    assert os.path.exists('/tmp/test.pkl')
    history.load('/tmp/test.pkl')
    assert abs(history[1].traj_len - 9 * np.sqrt(2)) < 1e-6
    assert len(history[1]) == 10
    assert len(history) == 3
    

def test_env_collision():
    '''场景碰撞检测逻辑测试'''    
    from rl.scripts.envs.assets.data.env_tools import CollisionChecker
    from rl.scripts.utils.tools import load_train_config
    filepath = "rl/configs/train_config.yaml"
    config = load_train_config(filepath)
    scene = config['scene_name']
    collision_checker = CollisionChecker(scene_config=config['scene_config_path'], scene=scene)
    assert collision_checker.is_collision(-7.5, -7.5) == True
    assert collision_checker.is_collision(-7.5, -7) == True


def test_config_data():
    """配置文件逻辑测试"""
    from rl.scripts.utils.tools import load_train_config
    filepath = "rl/configs/train_config.yaml"
    config = load_train_config(filepath)


def test_orientation_func():
    import numpy as np
    from rl.scripts.envs.utils import euler2quaternion, quaternion2euler
    yaw = np.deg2rad(45)
    roll = np.deg2rad(12)
    pitch = np.deg2rad(36)
    quaternion = euler2quaternion(roll=roll, pitch=pitch, yaw=yaw)
    roll_c, pitch_c, yaw_c = quaternion2euler(quaternion)
    assert abs(roll_c - roll) < 1e-5 and abs(pitch - pitch_c) < 1e-5 and abs(yaw - yaw_c) < 1e-5, "the function [euler2quaternion] or [quaternion2euler] don't work correctly!"
