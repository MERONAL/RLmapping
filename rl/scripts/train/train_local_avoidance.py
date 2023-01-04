import os
import argparse
import numpy as np
from functools import partial

from stable_baselines3 import SAC
# from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from rl.scripts.envs.env.local_env.local_nav_env import LocalNavContinusIncrementalEnv, DreamerEnv,LBULEnv, FixOrientationEnv, FrameStackEnv
from rl.scripts.envs.env.local_env.local_nav_env import SUCCESS, FAILED

env_list = {
    'LocalNavContinusIncrementalEnv': LocalNavContinusIncrementalEnv,
    'DreamerEnv': DreamerEnv,
    'LBULEnv': LBULEnv,
    'FixOrientationEnv': FixOrientationEnv,
    "FrameStackEnv":partial(FrameStackEnv, stack_frame=6)
}

def arg_parse():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config', type=str, default='rl/configs/train_config.yaml')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--log_path', type=str, default='~/exp/rl/local_avoidance')
    parser.add_argument('--env', type=str, default='FixOrientationEnv', choices=env_list.keys())
    args = parser.parse_args()
    return args

def train(args):
    log_path = os.path.expanduser(args.log_path)
    new_logger = configure(log_path, ['stdout','csv','tensorboard'])
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=0.1e6, save_path=os.path.join(log_path, 'model_pools'), name_prefix = 'rl_model')
    env = env_list[args.env](train_config=args.config,)
    env.eval_model()
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log='./tensorboard_log/',gamma=0.995, action_noise=action_noise,use_sde=True,use_sde_at_warmup=True,ent_coef="auto")
    # SAC.load(path = "/home/zy/exp/simple5_framestack10_26/model_pools/rl_model_2000000_steps.zip",env=env, force_reset=True)
    model.set_parameters(load_path_or_dict="/home/zy/exp/simple5_framestack10_26/model_pools/rl_model_2000000_steps.zip")
    model.set_logger(new_logger)
    model.learn(total_timesteps=int(10e6), callback=checkpoint_callback)
    model.save(os.path.join(log_path,'SAC_localconNav'))
    return model, env

def test(args, model=None, env=None) -> None:
    env = env_list[args.env](train_config=args.config, record_history=True) if env is None else env
    env.record_history = True
    log_path = os.path.expanduser(args.log_path)
    # model = SAC.load(os.path.join(log_path,'SAC_localconNav')) if model is None else model
    model = SAC.load(log_path) if model is None else model
    env.eval_model()
    obs = env.reset()
    test_times = 100
    res = np.zeros((test_times, ))
    index = 0
    episode_return = 0
    success_times = 0
    spl = 0
    ndgs = np.zeros((test_times, ))
    while index < test_times:
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        episode_return += rewards
        if dones:
            if info['status'] == SUCCESS:
                success_times += 1 
                spl += info['minimum_path_length'] / max(info['minimum_path_length'], info['real_path_length'])
            ndgs[index] = info['terminal2target_length'] / info['minimum_path_length']
            obs = env.reset()
            res[index] = episode_return
            episode_return = 0
            index += 1
    print("episode mean return :",res.mean()," std:", res.std())
    print(f"success times {success_times}, total simulation times: {test_times}, success rate:{success_times/test_times}, SPL:{spl / test_times}, NDG:{ndgs.mean()}")
    # env.save_history(file_path=os.path.join(log_path,'trajectory_data.pkl'))
    # env.close()

if __name__ == '__main__':
    args = arg_parse()
    if args.mode == 'train':
        model, env = train(args)
        test(args=args, model=model, env=env)
    else:    
        test(args)
