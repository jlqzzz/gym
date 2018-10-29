import os

import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

static_pose = np.array([
        5.81273309e-01,  1.85140196e+00,  9.87890988e-01,  6.92577659e-01,
        7.21321036e-01,  9.39957098e-04, -5.58881222e-03, -4.02314754e-01,
        9.24105695e-02, -1.99385973e-01,  5.09633304e-01, -8.10803600e-02,
       -3.88602482e-01, -2.68891183e-01,  3.53612888e-01, -4.25897305e-02,
       -1.63438418e-01,  5.24282219e-01, -1.37265252e-01, -4.25898086e-01,
       -1.59479603e-01, -1.32623473e-02, -3.00712543e-02,  2.17190240e-02,
        1.32694849e-02, -4.01792982e-02, -1.42574975e-02,  2.20017274e-02,
       -1.98734626e-02, -2.57005603e-02,  2.86861101e-01,  1.19334770e-01,
       -6.94627566e-03, -3.56228014e-01,  1.76173273e-01,  8.85846426e-02,
       -1.51822752e-01,  8.58910544e-02,  6.04917131e-02,  1.34360951e-16,
        1.53288495e-16,  1.62103376e+00, -1.34619146e-01, -7.16225699e-01,
        4.77384370e-01,  1.53709022e-01,  4.78019779e-01, -3.34083076e-01,
        1.24355058e-01,  9.99143514e-01,  1.25113980e-01,  1.34360951e-16,
        1.53288495e-16, -1.56958714e+00,  7.48583744e-02, -5.66177820e-01,
        3.85973918e-01,  2.73585494e-01,  3.40890608e-01, -2.53564494e-01,
        1.24355058e-01, -1.76736207e-01,  2.02803012e-01,  8.27758582e-02,
       -9.65568421e-02, -5.94256537e-02, -5.85218852e-02, -4.61241732e-01,
        2.85080917e-01, -3.52501891e-01,  8.03790108e-01,  2.95768088e-01,
       -2.35674041e-01, -4.20539202e-01, -8.86471446e-02, -2.55716844e-01,
       -3.82466155e-01,  4.78628789e-01, -5.67617355e-01,  1.24420896e+00,
       -2.91321278e-01, -1.46924282e-01, -1.01086691e+00, -4.87597454e-01,
        5.96239743e-02,  1.24702386e-01,  2.85313190e-02,  7.05861694e-02,
        1.15755165e-02,  3.04066133e-01,  4.05042642e-02, -3.90287354e-02,
       -1.78346137e-01,  5.76819600e-02, -2.22943366e-01,  7.56061385e-02,
        6.99746207e-02,  1.60217869e-01,  7.09537902e-02,  1.76059349e-02,
        8.51021297e-02,  8.45277392e-15, -1.99136100e-14,  2.78809996e-01,
       -9.28942162e-02, -2.01157708e-01, -5.29623515e-02,  3.81981214e-01,
        6.99803585e-01, -8.39942157e-02, -1.38777878e-15,  7.02151637e-01,
       -8.11115387e-02,  8.45277392e-15, -1.99136100e-14,  2.03576383e-01,
        3.90944269e-02,  1.07341884e-02,  3.26879671e-01, -6.37969194e-02,
       -1.22086384e+00,  6.61675860e-01, -1.38777878e-15, -1.18609850e+00,
        6.38141798e-01,
])

leftfoot_pose = np.array([
        5.68353339e-01, -1.09630962e+00,  9.58611477e-01,  7.21038299e-01,
        6.88121278e-01, -2.37813440e-02, -7.76218983e-02, -3.06130251e-01,
       -3.90444342e-03, -5.82455128e-01,  5.68612151e-01,  2.30539074e-01,
       -2.28148795e-01, -3.85431418e-01,  4.84978356e-01,  9.66249841e-02,
        4.59823024e-01,  2.71561363e-01,  5.94210894e-02, -1.50493084e-01,
       -4.63527540e-01,  1.13619437e-01,  3.13838953e-02,  1.35487937e-01,
        1.91628122e-02,  5.02769819e-02,  4.98934119e-02, -4.80526604e-02,
        2.55109825e-02, -1.94782464e-02, -2.15225521e-02,  2.41061080e-02,
       -2.41428017e-01, -3.80302400e-02,  3.58890988e-02, -4.72819558e-02,
       -6.43249449e-03,  1.69688993e-02,  2.60658876e-02,  1.61228579e-16,
       -1.26717817e-16,  1.22681791e+00,  3.26273703e-01, -9.61659517e-01,
        3.30602244e-01,  3.03300673e-01,  4.26270528e-01, -3.45703226e-01,
        1.24355058e-01,  9.47765244e-01,  1.13897549e-01,  1.61228579e-16,
       -1.26717817e-16, -1.36505467e+00,  2.16863948e-01, -1.60580257e-01,
        1.22907775e+00, -3.85797846e-01,  4.27018036e-01, -5.63014610e-01,
        1.24355058e-01, -9.49876437e-02, -9.59612030e-02,  2.21541196e-02,
       -1.99545308e+00, -2.55221903e-01,  1.95792410e-01, -5.07878759e-01,
        9.23599947e-01, -1.94850032e+00,  1.09745530e+00,  2.76444515e-01,
        2.64338634e+00, -2.20181845e-01,  1.57951807e+00,  6.21679306e-01,
       -1.03813785e+00,  1.09419591e+00, -1.14003238e-01,  3.96309293e+00,
        3.79316657e-01,  1.07498986e+00, -1.10010873e+00, -1.14861627e+00,
        4.41390050e-01, -1.00574631e+00, -3.77087109e-01,  3.97527641e-01,
        3.96562747e-03,  2.54181820e-01,  2.16583675e-01,  5.62246943e-01,
       -4.20956218e-01, -1.38457119e-01, -1.42359963e-01,  1.09846421e+00,
       -1.27670346e-01,  3.11221680e-01,  4.15542822e-01, -3.61151351e-02,
        1.03684498e-01,  2.51986595e-14,  2.53528562e-14, -1.68958576e+00,
        9.24299124e-01, -5.12584068e-01, -8.95842498e-01, -1.79682846e+00,
       -1.47065791e-01,  1.77192852e+00,  0.00000000e+00, -1.91827673e-01,
        1.71084471e+00,  2.51986595e-14,  2.53528562e-14,  2.29280146e-01,
       -5.22225089e-01, -6.54449987e-01,  1.35204265e+00,  9.15610509e-02,
       -7.88975248e-02, -2.26071613e-01,  0.00000000e+00, -7.34113722e-02,
       -2.18271648e-01,
])

rightfoot_pose = np.array([
        5.82383548e-01,  1.35492129e+00,  9.25901732e-01,  7.17197976e-01,
        6.92248997e-01,  2.60063013e-02,  7.57763846e-02, -4.80428057e-01,
       -6.27205011e-02,  4.22348735e-01,  2.12813232e-01,  1.68855718e-02,
       -2.32160206e-01, -2.83888275e-01,  2.82345404e-01,  3.26360862e-02,
       -5.78533014e-01,  4.72228500e-01, -2.35931863e-01, -2.96983480e-01,
       -2.11677022e-01, -6.79837160e-02, -5.80971239e-02,  1.52586457e-01,
        1.36632053e-02, -8.29366498e-02,  2.81340094e-02,  6.11350440e-02,
       -4.06644262e-02, -4.94015445e-02, -9.86109282e-02,  1.59635842e-02,
       -1.53436956e-01,  2.21473555e-02,  3.00755137e-02, -9.07658477e-02,
        2.41558804e-02,  1.67794034e-02, -3.69782909e-03,  3.46945275e-17,
       -5.26489041e-16,  1.46670838e+00, -2.59015588e-01, -3.14035347e-01,
        1.56050412e+00,  5.09246933e-01,  1.78400575e-01, -4.42927912e-01,
        1.24355058e-01,  7.01590943e-01,  2.00110725e-02,  3.46945275e-17,
       -5.26489041e-16, -1.48426465e+00, -2.47841990e-02, -9.82985143e-01,
        3.93941756e-01, -2.31823358e-01,  2.99325712e-01, -4.96766084e-01,
        1.24355058e-01, -2.23780881e-01, -3.19859511e-02, -4.93670608e-02,
       -1.79901015e+00, -1.48906674e-01, -5.02761646e-02, -1.30738569e-01,
       -5.55847036e-01,  7.76764636e-01, -2.76164863e-02,  1.09940352e+00,
        1.78844765e+00,  1.90539095e-02,  1.02971994e+00, -6.07744996e+00,
        5.43216920e-01,  1.67659330e+00,  4.33571352e-01,  1.83440420e+00,
       -3.81483856e-02,  6.89013528e-01,  4.09879471e+00,  5.45223920e-01,
       -2.87185610e-02,  4.00837398e-02,  2.38762730e-01,  1.66102124e-02,
       -5.29195526e-02, -5.21874070e-02,  8.06167720e-03, -1.02956273e-01,
        5.32830256e-01,  1.30265754e-01, -1.28190693e+00, -6.45289474e-01,
        1.99835589e-01,  6.84792077e-01, -2.86146055e-01,  8.15864508e-02,
        4.28602350e-01, -9.52724101e-15,  1.50332600e-14,  1.45402148e-01,
        1.94317231e-01, -8.21645658e-01,  6.45100814e-01,  4.76217187e-01,
        9.75095397e-02,  2.13909894e-01,  0.00000000e+00,  9.65119029e-02,
        2.06575898e-01, -9.52724101e-15,  1.50332600e-14,  9.93487979e-01,
       -1.13011328e+00,  6.94967642e-01, -1.69404432e+00,  1.79673428e+00,
        3.58216400e-01, -3.35993869e-01,  0.00000000e+00,  3.60750075e-01,
       -3.24438391e-01,
])

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))

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

    def reset_model(self, reset_type=None):
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

    def reset_model(self, reset_type=None):
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
        for ind in [22, 5, 29, 10, 17]:  # L/R hand/foot, head
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
        # standing
        head_height = self.sim.data.body_xpos[17, 2]
        head_upright = self.sim.data.body_xmat[17, 7]
        stand_reward = -0.5*np.abs(head_height-1.5) + head_upright
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus + stand_reward
        qpos = self.sim.data.qpos
        done = bool((head_height < 1.0) or (head_height > 2.0))
        state = self._get_obs()
        return state, reward, done, dict(reward_linvel=lin_vel_cost,
               reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus,
               reward_impact=-quad_impact_cost, img_ob=self.img_ob, show=head_upright)

    def reset_model(self, reset_type=None):
        if reset_type is not None:
            if reset_type == 'leftfoot_front':
                init_state = leftfoot_pose # np.load(os.path.join(os.path.dirname(__file__), "assets","leftfoot_front_state.npy"))
            elif reset_type == 'rightfoot_front':
                init_state = rightfoot_pose # np.load(os.path.join(os.path.dirname(__file__), "assets","rightfoot_front_state.npy"))
            elif reset_type == 'static':
                init_state = static_pose # np.load(os.path.join(os.path.dirname(__file__), "assets","static_state.npy"))
            else:
                assert(True, 'Wrong reset type')

            init_qpos = init_state[:63]
            init_qvel = init_state[63:]

            c = 0.01
            self.set_state(
                init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
                init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
            )
        else:
            c = 0.01
            self.set_state(
                self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
                self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
            )

        '''
        print('===reset_model===')
        print('self.init_qpos: ', self.init_qpos)
        print('self.init_qvel ',  self.init_qvel)
        '''

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
        for ind in [22, 5, 29, 10, 17]:  # L/R hand/foot, head
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
        # standing
        head_height = self.sim.data.body_xpos[17, 2]
        head_upright = self.sim.data.body_xmat[17, 7]
        stand_reward = -0.5*np.abs(head_height-1.5) + head_upright
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus + stand_reward
        qpos = self.sim.data.qpos
        done = bool((head_height < 1.0) or (head_height > 2.0))
        state = self._get_obs()
        return state, reward, done, dict(reward_linvel=lin_vel_cost,
               reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus,
               reward_impact=-quad_impact_cost, img_ob=self.img_ob, show=head_upright)

    def reset_model(self, reset_type=None):
        if reset_type is not None:
            if reset_type == 'leftfoot_front':
                init_state = leftfoot_pose # np.load(os.path.join(os.path.dirname(__file__), "assets","leftfoot_front_state.npy"))
            elif reset_type == 'rightfoot_front':
                init_state = rightfoot_pose # np.load(os.path.join(os.path.dirname(__file__), "assets","rightfoot_front_state.npy"))
            elif reset_type == 'static':
                init_state = static_pose # np.load(os.path.join(os.path.dirname(__file__), "assets","static_state.npy"))
            else:
                assert(True, 'Wrong reset type')

            init_qpos = init_state[:63]
            init_qvel = init_state[63:]

            c = 0.01
            self.set_state(
                init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
                init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
            )
        else:
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
