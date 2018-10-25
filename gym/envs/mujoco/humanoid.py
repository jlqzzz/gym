import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return np.sum(mass * xpos, 0) / np.sum(mass)

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.img_ob = np.zeros((128, 128, 3)).astype(np.uint8)
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)[0]
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[0]
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        state = self._get_obs()
        return state, reward, done, dict(reward_linvel=lin_vel_cost,
               reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus,
               reward_impact=-quad_impact_cost, img_ob=self.img_ob)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

class VisualHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        img = self.sim.render(width=128, height=128, depth=False, device_id=-1)
        return img

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)[0]
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[0]
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


class HumanoidCMUEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.img_ob = np.zeros((128, 128, 3)).astype(np.uint8)
        mujoco_env.MujocoEnv.__init__(self, 'humanoid_CMU.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        # return np.concatenate([data.qpos[7:],
        #                        data.body_xpos[17, [2]],
        #                        data.cinert.flat,
        #                        data.cvel.flat,
        #                        data.qfrc_actuator.flat,
        #                        data.cfrc_ext.flat])

        joint_angles = data.qpos[7:]
        head_height = data.body_xpos[17, [2]]
        torso_frame = data.body_xmat[14, :].reshape(3, 3)
        torso_pos = data.body_xpos[14, :]
        positions = []
        for ind in [22, 5, 29, 10]:
            torso_to_limb = data.body_xpos[ind] - torso_pos
            positions.append(torso_to_limb.dot(torso_frame))
        extremities = np.hstack(positions)
        torso_vertical_orientation = data.body_xmat[14, [6,7,8]]
        center_of_mass_velocity = data.subtree_linvel[14, :]
        velocity = data.qvel
        return np.concatenate([joint_angles,
                               head_height,
                               extremities,
                               torso_vertical_orientation,
                               center_of_mass_velocity,
                               velocity])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)[1]
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[1]
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = -0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        head_height = self.sim.data.body_xpos[17, 2]
        done = bool((head_height < 1.0) or (head_height > 2.0))
        state = self._get_obs()
        return state, reward, done, dict(reward_linvel=lin_vel_cost,
               reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus,
               reward_impact=-quad_impact_cost, img_ob=self.img_ob)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20


class HumanoidCMUSimpleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.img_ob = np.zeros((128, 128, 3)).astype(np.uint8)
        mujoco_env.MujocoEnv.__init__(self, 'humanoid_CMU_simple.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        # return np.concatenate([data.qpos[7:],
        #                        data.body_xpos[17, [2]],
        #                        data.cinert.flat,
        #                        data.cvel.flat,
        #                        data.qfrc_actuator.flat,
        #                        data.cfrc_ext.flat])

        joint_angles = data.qpos[7:]
        head_height = data.body_xpos[17, [2]]
        torso_frame = data.body_xmat[14, :].reshape(3, 3)
        torso_pos = data.body_xpos[14, :]
        positions = []
        for ind in [22, 5, 29, 10]:
            torso_to_limb = data.body_xpos[ind] - torso_pos
            positions.append(torso_to_limb.dot(torso_frame))
        extremities = np.hstack(positions)
        torso_vertical_orientation = data.body_xmat[14, [6,7,8]]
        center_of_mass_velocity = data.subtree_linvel[14, :]
        velocity = data.qvel
        return np.concatenate([joint_angles,
                               head_height,
                               extremities,
                               torso_vertical_orientation,
                               center_of_mass_velocity,
                               velocity])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)[1]
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[1]
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = -0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        head_height = self.sim.data.body_xpos[17, 2]
        done = bool((head_height < 1.0) or (head_height > 2.0))
        state = self._get_obs()
        return state, reward, done, dict(reward_linvel=lin_vel_cost,
               reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus,
               reward_impact=-quad_impact_cost, img_ob=self.img_ob)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
