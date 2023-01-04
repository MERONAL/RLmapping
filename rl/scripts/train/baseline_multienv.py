import os
import sys
import numpy as np
# from sklearn.model_selection import learning_curve
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from stable_baselines3 import PPO
from envs import LocalNavContinusEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.noise import NormalActionNoise
import os
tmp_path = "./result_ppo/"
new_logger = configure(tmp_path, ['stdout','csv','tensorboard'])

env = LocalNavContinusEnv()
# env = make_vec_env(LocalNavContinusEnv, n_envs=5)
# n_action = env.action_space.shape[0]
# action_noise = NormalActionNoise(mean=np.zeros(n_action),sigma=0.1 * np.ones(n_action))
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./tensorboard_log/',gamma=0.995)
model.set_logger(new_logger)
model.learn(total_timesteps=int(2e6))
model.save(os.path.join(tmp_path,'PPO_localconNav'))
del model
model = PPO.load(os.path.join(tmp_path,'PPO_localconNav'))
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
