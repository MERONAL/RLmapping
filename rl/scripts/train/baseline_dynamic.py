import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from stable_baselines3 import SAC
from envs import LocalDynamicNavContinusEnv2
from stable_baselines3.common.logger import configure
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.flatten_observation import FlattenObservation
tmp_path = "./result_contineous/SAC/"
# new_logger = configure(tmp_path, ['stdout','csv','tensorboard'])

env =FlattenObservation(FrameStack(LocalDynamicNavContinusEnv2(), num_stack=8))
# model = SAC("MlpPolicy", env, verbose=1, tensorboard_log='./tensorboard_log/',gamma=0.99)
# model.set_logger(new_logger)
# model.learn(total_timesteps=int(1e7))
# model.save(os.path.join(tmp_path,'SAC_localconNav'))
# del model
model = SAC.load(os.path.join(tmp_path,'SAC_localconNav'))
env.eval_model()
obs = env.reset()
test_times = 100
res = np.zeros((test_times, ))
index = 0

episode_return = 0
while index < test_times:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    episode_return += rewards
    if dones:
        obs = env.reset()
        res[index] = episode_return
        episode_return = 0
        index += 1
print("episode mean return :",res.mean()," std:", res.std())
