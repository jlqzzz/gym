import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, dict(show=reward)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self, init_state=None):
        if init_state:
            self.set_state(init_state[0], init_state[1])
        else:
            self.set_state(
                self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
                self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
            )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


class SparseWalker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        try:
            self.orig_pos
            self.cur_pos
        except:
            self.orig_pos = self.cur_pos = self.sim.data.qpos[0]
        posbefore = self.sim.data.qpos[0]
        try:
            self.do_simulation(np.clip(a, a_min=-1.0, a_max=1.0), self.frame_skip)
        except:
            ob = self.reset_model(None)
            return ob, 0, True, dict(show=0)
        self.cur_pos = self.sim.data.qpos[0]
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        if self.cur_pos - self.orig_pos >= 1.:
            sparse_reward = 1
            self.orig_pos = self.cur_pos
        else:
            sparse_reward = 0
        return ob, sparse_reward, done, dict(show=reward)
        # return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self, init_state=None):
        if init_state:
            self.set_state(init_state[0], init_state[1])
        else:
            self.set_state(
                self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
                self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
            )
        self.orig_pos = self.cur_pos = self.sim.data.qpos[0]
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
