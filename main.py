import gym
from final_project.envs.opt_env import OptEnv
import numpy as np

# env = gym.make('opt-env-v0')

from gym.spaces import Box
box = Box(low=-np.inf, high=np.inf, shape=(3,4))
print(box.sample())