import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register
from gymnasium.spaces import Box


class BedeSwimmerEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        obs_size = 8 if exclude_current_positions_from_observation else 10
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        xml_path = os.path.join(os.path.dirname(__file__), "bede_swimmer_native.xml")
        MujocoEnv.__init__(
            self, xml_path, 4, observation_space=observation_space, **kwargs
        )

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()
        x_velocity, y_velocity = (xy_position_after - xy_position_before) / self.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)
        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        info = {
            "reward_fwd": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        if self._exclude_current_positions_from_observation:
            position = position[2:]
        return np.concatenate([position, velocity]).ravel()

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()


register(
    id="BedeSwimmerNative-v0",
    entry_point="bede_swimmer_native:BedeSwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

