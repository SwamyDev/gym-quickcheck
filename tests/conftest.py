import numpy as np
import pytest

from tests.aux import until_done


@pytest.fixture(scope='session')
def gym_interface():
    return [('reset', ()), ('render', ()), ('step', (0,)), ('close', ())]


@pytest.fixture(scope='session')
def gym_properties():
    return ['action_space', 'observation_space']


@pytest.fixture
def make_observation_of():
    def obs_fac(obs_shape, agent_pos):
        obs = np.zeros(shape=obs_shape)
        obs[agent_pos] = 1
        return obs

    return obs_fac


@pytest.fixture
def capstdout(capsys):
    class _CapStdOut:
        def __init__(self, cap):
            self._cap = cap

        def read(self):
            return self._cap.readouterr()[0]

    return _CapStdOut(capsys)


@pytest.fixture
def sample_average_reward():
    def sample_average_reward_func(env, n):
        total = 0
        for _ in range(0, n):
            env.reset()
            total += sum(r for _, r, _, _ in until_done(env, env.action_space.sample))

        return total / n

    return sample_average_reward_func
