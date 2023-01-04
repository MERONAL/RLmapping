from re import T
import sys
import os
from turtle import st
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from envs import LocalNavEnv
import warnings
warnings.filterwarnings('ignore')

def test_local_env():
    from envs import LocalNavEnv
    env = LocalNavEnv()
    print(env.observation_space)
    state = env.reset()
    print(state)
    cum_reward = 0
    while True:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        # cum_reward += reward
        print(state)
        print(reward)
        if done:
            env.close()
            env.reset()
    env.close()
    print(cum_reward)




def test_local_continue_env():
    from envs import LocalNavContinusEnv
    env = LocalNavContinusEnv()
    print(env.observation_space)
    state = env.reset()
    print(state)
    cum_reward = 0
    while True:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print(state, reward)
        # cum_reward += reward
        if done:
            env.reset()



def test_local_continue_env2():
    from envs import LocalNavContinusEnv2
    env = LocalNavContinusEnv2()
    state = env.reset()
    print(state)
    cum_reward = 0
    while True:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print(env.previous_velocity)
        # print(state, reward)
        # cum_reward += reward
        if done:
            env.reset()

# test_local_continue_env2()


def test_local_dynamic_continue_env():
    from envs import LocalDynamicNavContinusEnv
    from gym.spaces import Box
    import numpy as np
    env = LocalDynamicNavContinusEnv()
    print(env.action_space)
    state = env.reset()
    cum_reward = 0
    while True:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        # print(state, reward)
        # cum_reward += reward
        if done:
            env.reset()

test_local_dynamic_continue_env()