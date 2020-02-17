import numpy as np
import pytest

from gym_quickcheck.envs.n_knob_env import NKnobEnv, ResetError
from tests.aux import assert_that, follows_contract, assert_obs_eq, until_done, unpack_done


@pytest.fixture
def env():
    return NKnobEnv()


@pytest.fixture
def idle():
    return make_action([0, 0, 0])


def make_action(values):
    return np.array([values])


def test_adherence_to_gym_contract(env, gym_interface, gym_properties):
    assert_that(env, follows_contract(gym_interface, gym_properties))


def test_initial_observation_is_direction_from_zero(env):
    env.seed(42)
    assert_obs_eq(env.reset(), make_obs([-1, -1, 1]))
    assert_obs_eq(env.reset(), make_obs([-1, 1, 1]))


def make_obs(values):
    return np.array(values)


def test_raise_error_when_stepping_without_resetting(env, idle):
    with pytest.raises(ResetError):
        env.step(idle)


def test_reward_range_is_from_negative_max_steps_to_minus_one(env):
    assert env.reward_range == (-env.max_len, -1)


def test_has_a_max_episode_length(env, idle):
    env.reset()
    assert sum(1 for _ in until_done(env, lambda: idle)) == env.max_len


def test_not_done_until_finished_or_reset(env, idle):
    env.reset()
    assert all(not unpack_done(env.step(idle)) for _ in range(env.max_len - 1))
    assert unpack_done(env.step(idle))
    env.reset()
    assert not unpack_done(env.step(idle))
