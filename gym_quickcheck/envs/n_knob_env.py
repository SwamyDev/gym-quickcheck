import sys

import gym
import numpy as np
from gym import spaces, utils
from gym.utils import seeding


class NKnobEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=7):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(n,), dtype=np.int8)
        self._num_knobs = n
        self._max_len = 200
        self._sensitivity = 0.01
        self._knobs = None
        self._elapsed_steps = None
        self._last_action = np.zeros((n,))
        self.seed()
        self.reward_range = (-self.max_len, -1)

    @property
    def max_len(self):
        return self._max_len

    @property
    def knobs(self):
        return self._knobs

    @property
    def sensitivity(self):
        return self._sensitivity

    def seed(self, seed=None):
        # noinspection PyAttributeOutsideInit
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._knobs = self.np_random.rand(self._num_knobs) * 2 - np.ones(self._num_knobs)
        self._elapsed_steps = 0
        return self._make_observation(self._knobs)

    def _make_observation(self, direction):
        is_set_correctly = (direction > self.sensitivity) | (direction < -self.sensitivity)
        return ((direction > 0) * 2 - np.ones_like(direction)) * is_set_correctly

    def step(self, action):
        if self._knobs is None:
            raise ResetError("Cannot call env.step() before calling reset()")

        self._elapsed_steps += 1
        obs = self._make_observation(self._knobs - action)
        self._last_action = action

        has_max_len = self._elapsed_steps == self.max_len
        is_finished = (obs == np.zeros_like(obs)).all()
        return obs, -1, has_max_len or is_finished, None

    def render(self, mode='human'):
        s = []
        for k, a in zip(self.knobs, self._last_action):
            color = 'green' if abs(k - a) < self.sensitivity else 'red'
            s.append(utils.colorize(f"({k:.3f}/{a:.3f})", color=color, highlight=True))
        sys.stdout.write(f"{' '.join(s)}\n")


class ResetError(RuntimeError):
    pass
