from gym import Env
from gym.spaces import Box, MultiBinary
import numpy as np
from dataclasses import dataclass


@dataclass
class IterationObs:
    x: np.ndarray
    g: np.ndarray
    obj: float

@dataclass
class Obs:
    prev_it: IterationObs
    cur_it: IterationObs
    d: np.ndarray


def obs_arr_to_object(obs, N):
    prev_it = IterationObs(
        x=obs[:N], g=obs[N:2*N], obj=obs[2*N])

    i = 2*N+1
    cur_it = IterationObs(
        x=obs[i:i+N], g=obs[i+N:i+2*N], obj=obs[i+2*N])

    d = obs[2*(2*N+1):]

    obs_object = Obs(prev_it=prev_it, cur_it=cur_it, d=d)
    return obs_object

def obs_object_to_arr(obs, N):
    obs_arr = np.zeros(2*(2*N+1)+N)
    obs_arr[0:2*N+1] = np.concatenate([obs.prev_it.x, obs.prev_it.g, [obs.prev_it.obj]])
    obs_arr[2*N+1:2*(2*N+1)] = np.concatenate([obs.cur_it.x, obs.cur_it.g, [obs.cur_it.obj]])
    obs_arr[2*(2*N+1):] = obs.d
    return obs_arr


class OptEnv(Env):
    def __init__(self, N, eps):
        self.N = N
        self.eps = eps
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2*(2*N+1)+N,))
        # self.action_space = Box(low=.95, high=1.05, shape=(N,))
        self.action_space = MultiBinary(N)

        A = np.random.randn(self.N, self.N)
        A = A.T@A
        self.f = lambda x: .5*x@(A@x)
        self.df = lambda x: A@x

        self.i = 0
        self.max_iter_mean = 1
        self.mode = 'train'


    def set_mode(self, mode):
        self.mode = mode


    def reset(self):
        self.j = 0

        self.i += 1

        if self.mode == 'train':
            if self.i % 1e3 == 0:
                self.max_iter_mean += 1
            self.max_iter = np.random.geometric(1/self.max_iter_mean)
            self.max_iter = max(1, self.max_iter)
        else:
            self.max_iter = 500

        d = 0.1*np.ones(self.N)

        x_prev=5*np.ones(self.N)
        prev_it = IterationObs(
            x=x_prev,
            g=self.df(x_prev),
            obj=self.f(x_prev)
        )

        x_cur = prev_it.x - d*prev_it.g
        cur_it = IterationObs(
            x=x_cur,
            g=self.df(x_cur),
            obj=self.f(x_cur)
        )

        obs_object = Obs(prev_it=prev_it, cur_it=cur_it, d=d)
        self.obs = obs_object_to_arr(obs_object, self.N)
        return self.obs

    
    def step(self, action):
        d_scale = np.zeros(self.N)
        for i in range(self.N):
            d_scale[i] = 1.05 if action[i] == 1 else 1/1.05

        if self.i % 1e3 == 0:
            print(d_scale)

        obs = obs_arr_to_object(self.obs, self.N)

        new_d = obs.d * d_scale

        x = obs.prev_it.x - new_d*obs.prev_it.g
        g = self.df(x)

        cur_it = IterationObs(
            x=x,
            g=g,
            obj=self.f(x)
        )

        
        obs = Obs(prev_it=obs.cur_it, cur_it=cur_it, d=new_d)
        self.obs = obs_object_to_arr(obs, self.N)

        g_normsq = g@g


        # reward = -cur_it.obj -10*np.log(g_normsq) + min(0, 5*np.log(np.linalg.norm(new_d))) - 5*np.log(self.j+1)
        reward = -cur_it.obj # - 1/np.linalg.norm(new_d)
        # if (new_d < 1e-6).any() or g_normsq >= 1e11:
        #     reward -= 100

        # if g_normsq <= self.eps**2:
        #     reward += 20

        notes = []
        if g_normsq >= 1e11:
            notes += ['gradient norm too large']

        if (new_d < 1e-6).any():
            notes += ['norm of d too small']

        if self.j >= self.max_iter:
            notes += ['too many iterations']

        if g_normsq <= self.eps**2:
            notes += ['success!']


        if g_normsq >= 1e11 or (new_d < 1e-6).any() or self.j >= self.max_iter:
            done = True
        elif g_normsq <= self.eps**2:
            done = True
        else:
            done = False

        self.j += 1

        return self.obs, reward, done, {'notes':notes}