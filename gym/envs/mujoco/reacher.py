import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])



class ReacherObsDoneEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_obs.xml', 2)

    def _step(self, a):
        # collision detect
        is_collision = False
        if self.unwrapped.data.ncon > 0:
            geo1 = self.unwrapped.data.obj.contact[0].geom1
            geo2 = self.unwrapped.data.obj.contact[0].geom2
            if geo1 == 10 or geo2 == 10 or geo1 == 11 or geo2 == 11:
                is_collision = True
        else:
            obs1x, obs1y = self.get_body_com("obs1")[:2]
            obs2x, obs2y = self.get_body_com("obs2")[:2]
            if obs1x != 0 or obs1y != 0.15 or obs2x != 0 or obs2y != -0.15:
                is_collision = True

        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward_collsion = -100. if is_collision else 0
        alive_bonus = 0. if not is_collision else 0.
        reward = reward_dist + reward_ctrl + reward_collsion + alive_bonus
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = is_collision
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.init_qpos
        qpos[:-4] = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq-4) + self.init_qpos[:-4]
        while True:
            self.goal_x = self.np_random.uniform(low=-.2, high=0, size=1)
            self.goal_y = self.np_random.uniform(low=-.2, high=.2, size=1)
            if np.linalg.norm([self.goal_x, self.goal_y]) < 2:
                break
        qpos[-6:-4] = [self.goal_x, self.goal_y]
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-6:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:], # target and obstacle pos 
            self.model.data.qvel.flat[:2], # arm end point speed only 
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])