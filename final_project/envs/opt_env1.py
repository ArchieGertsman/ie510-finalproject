from gym import Env
from gym.spaces import Box, Discrete
import numpy as np


class OptEnv1(Env):
    '''
    Observation space:
    - learning rate
    - prev grad norm
    - prev objective value
    - curr grad norm
    - curr objective value
    - s@s
    - y@y
    - s@y

    Action space:
    - 0 (lr *= 1/1.05) or 1 (lr *= 1.05)
    '''

    def __init__(self, f, grad_f, x0, lr0):
        self.observation_space = Box(
            low=np.array([0, 0, 0, np.NINF, np.NINF, 0, 0, np.NINF]), 
            high=np.inf
        )
        self.action_space = Discrete(2)

        self.f = f
        self.grad_f = grad_f
        self.x0 = x0
        self.lr0 = lr0
        self.mode = 'train'
        self.max_iter_mean = 1
        self.eps = 1e-3
        self.n_episodes = 0


    def set_mode(self, mode):
        self.mode = mode


    def reset(self):
        self.n_episodes += 1
        self.n_iter = 0

        if self.mode == 'train':
            if self.n_episodes % 1e3 == 0:
                self.max_iter_mean += 3
            self.max_iter = np.random.geometric(1/self.max_iter_mean)
            self.max_iter = max(1, self.max_iter)
        else:
            self.max_iter = 500

        self.lr = self.lr0
        self.x_curr = self.x0.copy()
        self.g_curr = self.grad_f(self.x0)
        self.g_norm_curr = np.linalg.norm(self.g_curr)
        self.obj_val_curr = self.f(self.x0)

        self._update_iterate()
        self._update_grad()
        self._update_obj_val()
        self._update_g_norm()

        return self._observe()


    def step(self, action):
        lr_scale = 1.05 if action == 1 else 1/1.05
        self.lr *= lr_scale

        self._update_iterate()
        self._update_grad()
        self._update_obj_val()
        self._update_g_norm()
        
        obs = self._observe()
        reward = -self.obj_val_curr
        if self.lr >= .6:
            reward -= 8*self.lr**2
        elif self.lr < .005:
            reward += np.log(self.lr + 1e-8)
        done = self._check_if_done()
        
        self.n_iter += 1

        return obs, reward, done, {}


    def _observe(self):
        s = self.x_curr - self.x_prev
        y = self.g_curr - self.g_prev

        return np.array([
            self.lr,
            self.g_norm_prev,
            self.obj_val_prev,
            self.g_norm_curr,
            self.obj_val_curr,
            s@s,
            y@y,
            s@y
        ])


    def _check_if_done(self):
        return self.g_norm_curr <= self.eps or \
            self.n_iter > self.max_iter or \
            self.g_norm_curr >= 1e11

    
    def _update_iterate(self):
        x_new = self.x_curr - self.lr * self.g_curr
        self.x_prev = self.x_curr
        self.x_curr = x_new

    def _update_grad(self):
        self.g_prev = self.g_curr
        self.g_curr = self.grad_f(self.x_curr)

    def _update_obj_val(self):
        self.obj_val_prev = self.obj_val_curr
        self.obj_val_curr = self.f(self.x_curr)

    def _update_g_norm(self):
        self.g_norm_prev = self.g_norm_curr
        self.g_norm_curr = np.linalg.norm(self.g_curr)