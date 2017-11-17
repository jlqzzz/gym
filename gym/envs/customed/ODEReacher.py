#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : ODEReacher.py
# Purpose :
# Creation Date : 17-11-2017
# Last Modified : 2017年11月17日 星期五 22时32分34秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math
import logging
import gym
from gym import spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)


class ODEReacherEnv(gym.Env):
    def __init__(self):
        self.state_dim = 5
        self.action_dim = 3
        #self.action_space = spaces.Discrete(self.action_dim)
        self.action_space = spaces.Box(-0.2, 0.2, (self.action_dim, ))
        self.observation_space = spaces.Box(-2, 2, (self.state_dim,))
        self._seed()

        self.dt = 0.1
        self.__current_reward = None
        self.__current_state = None
        self.__trajectory_state = None
        self.__counter = 0
        self.target = np.ones(2, dtype=np.float32)
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        act = np.clip(action, a_min=-0.2, a_max=0.2)
        # act = np.zeros((3,), np.float32)
        # if action == 0:
        #     act[0] = -0.1
        # elif action == 1:
        #     act[0] = 0.1
        # elif action == 2:
        #     act[1] = -0.1
        # elif action == 3:
        #     act[1] = 0.1
        # elif action == 4:
        #     act[2] = -0.1
        # elif action == 5:
        #     act[2] = 0.1

        next_state, reward = self.do_action(act)
        self.__counter += 1
        return next_state, reward, (self.__counter >= 100), {}

    def _reset(self):
        self.__current_state = np.zeros((self.state_dim,), np.float32)
        self.__current_state[-2:] = self.target
        self.__current_reward = 0
        self.__trajectory_state = np.zeros((3,), np.float32)
        return self.get_current_state()

    def _render(self, mode='human', close=False):
        #show_trajectory([self.__trajectory_state], -1)
        #plt.plot(self.target[0], self.target[1], 'o')
        #plt.pause(self.dt)
        pass

    def get_current_state(self):
        return self.__current_state

    def get_current_reward(self):
        return 2.-np.linalg.norm(self.target - self.__current_state[:2])

    def do_action(self, a):
        a = np.asarray(a, np.float32)
        self.__trajectory_state += a
        self.__current_state[:-2] = np.asarray(forward_kinematic(self.__trajectory_state), np.float32)
        self.__current_state[-2:] = self.target
        return self.get_current_state(), self.get_current_reward()

    @staticmethod
    def __ode_func(x, t):
        raise NotImplementedError('')
        # v = [0 1] * [p] + [0]
        # a   [a b]   [v]   [c]
        dx = np.zeros_like(x)
        for i, xi in enumerate(x):
            pass


def visualize(thetas):
    xs = (0.0, math.cos(thetas[0]), math.cos(thetas[0]) - math.cos(thetas[0] + thetas[1]),
          math.cos(thetas[0]) - math.cos(thetas[0] + thetas[1]) + 0.5 * math.cos(thetas[0] + thetas[1] + thetas[2]))
    ys = (0.0, math.sin(thetas[0]), math.sin(thetas[0]) - math.sin(thetas[0] + thetas[1]),
          math.sin(thetas[0]) - math.sin(thetas[0] + thetas[1]) + 0.5 * math.sin(thetas[0] + thetas[1] + thetas[2]))
    plt.cla()
    plt.axis([-3, 3, -3, 3])
    plt.plot(xs, ys)


def show_trajectory(traj, delay=1.0):
    plt.ion()
    for t in traj:
        visualize(t)
        plt.draw()
        if delay >= 0.:
            plt.pause(delay)


def forward_kinematic(thetas):
    theta = math.pi - thetas[0] - thetas[1] - thetas[2]
    x = math.cos(thetas[0]) - math.cos(thetas[0] + thetas[1]) + 0.5 * math.cos(thetas[0] + thetas[1] + thetas[2])
    y = math.sin(thetas[0]) - math.sin(thetas[0] + thetas[1]) + 0.5 * math.sin(thetas[0] + thetas[1] + thetas[2])
    return x, y, theta


def inverse_kinematic(x, y, theta, p0=None):
    target = np.asarray((x, y, theta), dtype=np.float32)
    if p0 is None:
        p0 = np.zeros((3,), dtype=np.float32)

    def target_func(p):
        FK = np.asarray(forward_kinematic(p), dtype=np.float32)
        return np.sum((FK - target)**2)
    return opt.fmin(target_func, p0, disp=0)


def main():
    x, y = 2.75, 1.0
    while 1:
        visualize(inverse_kinematic(x, y, math.pi))
        x, y = plt.ginput(1)[0]


if __name__ == "__main__":
    main()

