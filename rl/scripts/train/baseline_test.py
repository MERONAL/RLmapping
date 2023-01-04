import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from stable_baselines3 import SAC
from envs import MultiframeLocalNavContinusEnv
from stable_baselines3.common.logger import configure
tmp_path = "./simple5-world-frame-stack"
new_logger = configure(tmp_path, ['stdout','csv','tensorboard'])


env = MultiframeLocalNavContinusEnv()
# env.eval_model()
# env = make_vec_env(LocalNavContinusEnv, n_envs=5)
# n_action = env.action_space.shape[0]
# # action_noise = NormalActionNoise(mean=np.zeros(n_action),sigma=0.1 * np.ones(n_action))
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log='./tensorboard_log/',gamma=0.995)
model.set_logger(new_logger)
model.learn(total_timesteps=int(3e6))
model.save(os.path.join(tmp_path,'SAC_localconNav'))
del model

model = SAC.load(os.path.join(tmp_path,'SAC_localconNav'))
env.eval_model()
obs = env.reset()
test_times = 100
res = np.zeros((test_times, ))
index = 0
episode_return = 0
success_times = 0
spl = 0
while index < test_times:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    episode_return += rewards
    if dones:
        if rewards > 10:
             success_times += 1 
             spl += info['mini_path'] / max(info['mini_path'], info['path_length'])

        obs = env.reset()
        res[index] = episode_return
        episode_return = 0
        index += 1
print("episode mean return :",res.mean()," std:", res.std())
print(f"success times {success_times}, total simulation times: {test_times}, success rate:{success_times/test_times}, SPL:{spl / test_times}")
