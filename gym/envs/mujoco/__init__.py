from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv, SparseAntEnv
from gym.envs.mujoco.ant_distance import AntDistEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv, SparseHalfCheetahEnv, TwoHalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv, SparseHopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv, SparseWalker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv, SparseHumanoidEnv, VisualHumanoidEnv, HumanoidCMUEnv, HumanoidCMUSimpleEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv, SparseInvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv, SparseInvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv, SparseReacherEnv, RobustReacherEnv, VisualReacherEnv, BaxterReacherEnv, BaxterRightReacherEnv, BaxterLeftReacherEnv, UR5ReacherEnv, UR5ReacherAccEnv, ReacherPosEnv, ReacherDoneEnv, TwoReacherEnv, ReacherObsDoneEnv, ReacherSpeedEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv, SparseHumanoidStandupEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv
