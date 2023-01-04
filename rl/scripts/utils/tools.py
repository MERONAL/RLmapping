import os
import yaml

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_root_work_path() -> str:
    """获取ros的工作目录，会得到一个到src文件夹的绝对路径
    Returns:
        str: 工作目录的绝对路径
    """
    import os
    try:
        ros_package_path = os.environ['ROS_PACKAGE_PATH']
    except:
        raise Exception('ROS_PACKAGE_PATH is not set, please 【source ros workspace】')
    return ros_package_path.split(':')[0]


def _add_abs_path(config:dict, abs_path:str):
    """将配置文件中的相对路径转换为绝对路径

    Args:
        config (dict): 原始配置文件
        abs_path (str): 工作绝对路径

    Returns:
        _type_: dict
    """
    for key, item in config.items():
        if isinstance(item, dict):
            config[key] = _add_abs_path(config[key], abs_path)
        elif 'path' in key and isinstance(item, str):
            config[key] = os.path.join(abs_path, item)
            if config[key].endswith('.yaml'):
                config[key] = load_yaml(config[key])
    return config

def load_train_config(path):
    """为强化学习训练加载配置文件，包括gazebo相关环境的参数和训练参数;
    Args:
        path (str): 配置文件的相对路径
    Returns:
        dict: 配置文件的内容
    """
    work_root_path = get_root_work_path()
    if not path.startswith('/'):
        path = os.path.join(work_root_path, path)
    config = load_yaml(path)
    config = _add_abs_path(config, work_root_path)
    # 将gazebo场景模型信息补充完整
    config['gazebo']['world_file_path'] = config['gazebo']['world_file_path'].format(config['scene_name'])
    return config   
    