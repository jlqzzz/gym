import numpy as np
from scipy.stats import logistic
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")

        reward_addon = 0
        reward_dist = - np.linalg.norm(vec)
        # reward_dist = - (logistic.cdf(np.linalg.norm(vec)/0.3)-0.5)*10 # np.clip(1.0/np.linalg.norm(vec), a_min = 0, a_max=50)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl + reward_addon
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, show=reward - reward_addon)

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


class ReacherPosEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_pos.xml', 2)
        # self.action_space = spaces.Box(low, high)
        # self.action_space = spaces.Discrete(5) # ll lr hl hr stop

    def _step(self, a):
        # a should be in (5/180)*pi radius
        ctl_angle_limit = (5/180)*np.pi
        # if a == 0:
        #     action = np.array([ctl_angle_limit, 0])
        # elif a == 1:
        #     action = np.array([-ctl_angle_limit, 0])
        # elif a == 2:
        #     action = np.array([0, ctl_angle_limit])
        # elif a == 3:
        #     action = np.array([0, -ctl_angle_limit])
        # elif a == 4:
        #     action = np.array([0, 0])

        a = np.clip(a, a_min=-ctl_angle_limit, a_max=ctl_angle_limit)
        
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        vec_to_origin = self.get_body_com("fingertip")

        reward_addon = 0 #np.linalg.norm(vec_to_origin)*0.8

        reward_dist = - np.linalg.norm(vec)
        # reward_dist = - (logistic.cdf(np.linalg.norm(vec)/0.3)-0.5)*10
        reward_ctrl = - np.square(a).sum()
        # reward_ctrl = 0
        reward = reward_dist + reward_ctrl + reward_addon

        action = a
        ctl = self.model.data.qpos.flat[:2] + action
        self.do_simulation(ctl, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, show=reward-reward_addon)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2 and np.linalg.norm(self.goal) > 0.1:
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


class ReacherDoneEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = 1. / np.linalg.norm(vec) # np.clip(1.0/np.linalg.norm(vec), a_min = 0, a_max=50)
        reward_dist_clip = 50
        if reward_dist > reward_dist_clip:
            reward_dist = reward_dist_clip + np.log(reward_dist - reward_dist_clip)

        reward_ctrl = - np.square(a).sum()
        alive_penlty = 0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        threshold = 0.03
        done = False
        reward_done = 0
        reward = reward_ctrl + alive_penlty + reward_done + reward_dist
        return ob, reward, done, dict(alive_penlty=alive_penlty, reward_ctrl=reward_ctrl, reward_done=reward_done)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 1.8:
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


class TwoReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'two_reacher.xml', 2)

    def _step(self, a):
        vec1 = self.get_body_com("fingertip") - self.get_body_com("target1")
        reward_dist1 = 1. / np.linalg.norm(vec1) # np.clip(1.0/10*np.linalg.norm(vec1), a_min = 0, a_max=50)#
        vec2 = self.get_body_com("fingertip") - self.get_body_com("target2")
        reward_dist2 = 1. / np.linalg.norm(vec2) # np.clip(1.0/10*np.linalg.norm(vec2), a_min = 0, a_max=50)#
        reward_dist_clip = 30
        if reward_dist1 > reward_dist_clip:
            reward_dist1 = reward_dist_clip + np.log(reward_dist1 - reward_dist_clip)
        if reward_dist2 > reward_dist_clip: 
            reward_dist2 = reward_dist_clip + np.log(reward_dist2 - reward_dist_clip)

        reward_ctrl = - np.square(a).sum()
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        threshold = 0.03
        # done = np.linalg.norm(vec1) < threshold or np.linalg.norm(vec2) < threshold
        done = False
        penalty_alive = 0
        reward_done  = 0 # = 30 if done else 0
        reward = reward_dist1 + reward_dist2 + reward_ctrl + penalty_alive + reward_done
        reward_show = reward_dist1 + reward_ctrl + penalty_alive + reward_done
        return ob, reward, done, dict(reward_dist1=reward_dist1, reward_dist2=reward_dist2, reward_ctrl=reward_ctrl, show=reward_show)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal1 = self.np_random.uniform(low=-.2, high=.2, size=2)
            self.goal2 = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal1) < 2 and np.linalg.norm(self.goal2) < 2 \
                and np.linalg.norm(self.goal1 - self.goal2) > 0.15:
                break
        qpos[-4:-2] = self.goal1
        qpos[-2:] = self.goal2
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target1"),
            self.get_body_com("fingertip") - self.get_body_com("target2")
        ])


class ReacherSpeedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward_penalty = -5.
        reward = reward_dist + reward_ctrl + reward_penalty
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = True if reward_dist < 0.02 else False
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
        else:  # or if obstacles are moving(not locate at origin places)
            obs1x, obs1y = self.get_body_com("obs1")[:2]
            obs2x, obs2y = self.get_body_com("obs2")[:2]
            if obs1x != 0 or obs1y != 0.19 or obs2x != 0 or obs2y != -0.19:
                is_collision = True

        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        
        reward_ctrl = - np.square(a).sum()
        reward_collsion = -30. if is_collision else 0
        alive_bonus = 0. if not is_collision else 0.
        reward = reward_dist + reward_ctrl + reward_collsion + alive_bonus
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs ()
        done = is_collision
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.init_qpos
        qpos[:-4] = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq-4) + self.init_qpos[:-4]
        while True:
            #self.goal_x = self.np_random.uniform(low=-.2, high=0, size=1)
            #self.goal_y = self.np_random.uniform(low=-.2, high=.2, size=1)
            self.goal_x, self.goal_y = [0, .15]
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

##########################################################
##########################################################

class UR5ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'ur5/ur5.xml', 2)
        self.dist = np.linalg.norm(self.get_body_com("ee_link")-self.get_body_com("target"))
        self.pre_dist = 0

    def _step(self, a):
        # origin version
        vec = self.get_body_com("ee_link")-self.get_body_com("target")
        
        reward_addon = 0
        reward_dist = - np.linalg.norm(vec) # np.clip(1.0/np.linalg.norm(vec), a_min = 0, a_max=50)
        reward_ctrl = 0 # - np.square(a).sum()
        reward = reward_dist + reward_ctrl + reward_addon
        self.do_simulation(a, self.frame_skip)
        
        print(self.unwrapped.data.obj.contact[0].geom1, self.unwrapped.data.obj.contact[0].geom2)


        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, show=reward-reward_addon)

        # # second version
        # self.pre_dist = self.dist
        # self.dist = np.linalg.norm(self.get_body_com("ee_link")-self.get_body_com("target"))
        # reward_dist = (self.pre_dist - self.dist) - self.dist*10
        # reward_hit = 

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 3.0

    def reset_model(self):
        # ur5 max length=1
        qpos = self.init_qpos
        while True:
            self.goal_x = self.np_random.uniform(low=-0.8, high=-0.5)
            self.goal_y = self.np_random.uniform(low=-0.5, high=0.5)
            if abs(self.goal_y) < 0.2:
                continue
            self.goal_z = self.np_random.uniform(low=0.2, high=0.7)
            self.goal = np.array([self.goal_x, self.goal_y, self.goal_z])
            if np.linalg.norm(self.goal) < 0.7:
                break
        qpos[:-3] = np.array([0, -1.1, 2.1, -1.0, 1.40, 0.76])  # critical
        qpos[-3:] = self.goal
        self.set_state(qpos, self.init_qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:-3]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[-3:],
            self.model.data.qvel.flat[:-3],
            self.get_body_com("ee_link") - self.get_body_com("target")
        ])


class UR5ReacherPosEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'ur5/ur5_pos.xml', 2)

    def _step(self, a):
        # a should be in (5/180)*pi radius
        ctl_angle_limit = (5/180)*np.pi

        a = np.clip(a, a_min=-ctl_angle_limit, a_max=ctl_angle_limit)
        
        vec = self.get_body_com("ee_link")-self.get_body_com("target")
        vec_to_origin = self.get_body_com("ee_link")

        reward_addon = 0 # np.linalg.norm(vec_to_origin)*0.1

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        # reward_ctrl = 0
        reward = reward_dist + reward_ctrl + reward_addon

        action = a
        ctl = self.model.data.qpos.flat[:-3] + action
        self.do_simulation(ctl, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, show=reward-reward_addon)


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 3.0


    def reset_model(self):
        # ur5 max length=1
        qpos = self.init_qpos
        while True:
            self.goal_x = self.np_random.uniform(low=-0.8, high=-0.5)
            self.goal_y = self.np_random.uniform(low=-0.5, high=0.5)
            if abs(self.goal_y) < 0.2:
                continue
            self.goal_z = self.np_random.uniform(low=0.2, high=0.7)
            self.goal = np.array([self.goal_x, self.goal_y, self.goal_z])
            if np.linalg.norm(self.goal) < 0.7:
                break
        qpos[:-3] = np.array([0, -1.1, 2.1, -1.0, 1.40, 0.76])  # critical
        qpos[-3:] = self.goal
        self.set_state(qpos, self.init_qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:-3]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[-3:],
            self.model.data.qvel.flat[:-3],
            self.get_body_com("ee_link") - self.get_body_com("target")
        ])
