from .mujoco_env import MujocoEnv
from .sumo import SumoEnv

from gym.envs.registration import register

register(
    id="RoboSumo-Ant-vs-Ant-v0",
    entry_point="robosumo.envs.sumo:SumoEnv",
)

print("✅ RoboSumo environment registered!")
