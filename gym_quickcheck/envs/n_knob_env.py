import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class NKnobEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=3):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(n,), dtype=np.int8)
        self._num_knobs = n
        self._max_len = 200
        self._knobs = None
        self._elapsed_steps = None
        self.seed()
        self.reward_range = (-self.max_len, -1)

    @property
    def max_len(self):
        return self._max_len

    def seed(self, seed=None):
        # noinspection PyAttributeOutsideInit
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._knobs = self.np_random.rand(self._num_knobs) * 2 - np.ones(self._num_knobs)
        self._elapsed_steps = 0
        return (self._knobs > 0) * 2 - np.ones_like(self._knobs)

    def step(self, action):
        if self._knobs is None:
            raise ResetError("Cannot call env.step() before calling reset()")

        self._elapsed_steps += 1
        return None, None, self._elapsed_steps == self.max_len, None

    def render(self, mode='human'):
        pass


class ResetError(RuntimeError):
    pass
