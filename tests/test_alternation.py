import numpy as np
import pytest
from gym import utils
from more_itertools import last
from pytest import approx

from gym_quickcheck.envs.alteration_env import AlternationEnv
from tests.aux import assert_that, follows_contract, assert_obs_eq, unpack_reward, unpack_obs, until_done, \
    run_example


@pytest.fixture
def env():
    return AlternationEnv()


@pytest.fixture
def obs_shape(env):
    return env.observation_space.shape


@pytest.fixture
def make_observation(obs_shape, make_observation_of):
    def obs_fac(agent_pos):
        return make_observation_of(obs_shape, agent_pos)

    return obs_fac


@pytest.fixture
def sample_reset(env, obs_shape):
    def sample_reset_func(n):
        total = np.zeros(obs_shape)
        for _ in range(n):
            total += env.reset()
        return total / n

    return sample_reset_func


@pytest.fixture
def alternate_right_left():
    direction = 0

    def alternate():
        nonlocal direction
        direction = (direction + 1) % 2
        return direction

    return alternate


@pytest.fixture
def make_state_string():
    def state_fac(agent_pos, reward=None):
        s = ['#', '#']
        if reward is None:
            color = 'gray'
        elif reward:
            color = 'green'
        else:
            color = 'red'
        s[agent_pos] = utils.colorize(s[agent_pos], color=color, highlight=True)
        return "".join(s)

    return state_fac


def test_adherence_to_gym_contract(env, gym_interface, gym_properties):
    assert_that(env, follows_contract(gym_interface, gym_properties))


def test_agent_starts_randomly_left_or_right(sample_reset):
    avg_obs = sample_reset(10000)
    assert left(avg_obs) == approx(0.5, rel=0.1) and right(avg_obs) == approx(0.5, rel=0.1)


def left(obs):
    return obs[0]


def right(obs):
    return obs[1]


def test_alternate_the_agent_position(env, make_observation):
    force_reset(env, left)
    assert_obs_eq(unpack_obs(env.step(go_right())), make_observation(agent_pos=1))
    force_reset(env, right)
    assert_obs_eq(unpack_obs(env.step(go_left())), make_observation(agent_pos=0))


def force_reset(env, pos):
    obs = env.reset()
    while pos(obs) != 1:
        obs = env.reset()
    return obs


def go_right():
    return 0


def go_left():
    return 1


def test_not_alternating_does_not_change_agent_position(env, make_observation):
    force_reset(env, left)
    assert_obs_eq(unpack_obs(env.step(go_left())), make_observation(agent_pos=0))
    force_reset(env, right)
    assert_obs_eq(unpack_obs(env.step(go_right())), make_observation(agent_pos=1))


def test_environment_is_done_after_episode_length_is_reached(env):
    env.reset()
    assert sum(1 for _ in until_done(env, 0)) == env.len_episode


def test_alternating_position_gives_reward(env, alternate_right_left):
    force_reset(env, left)
    total_reward = sum(unpack_reward(t) for t in until_done(env, alternate_right_left))
    assert total_reward == approx(env.reward_range[1], rel=env.reward.std * 6)


def test_keeping_at_the_same_position_causes_penalties(env):
    force_reset(env, left)
    total_penalty = sum(unpack_reward(t) for t in until_done(env, go_left()))
    assert total_penalty == approx(env.reward_range[0], rel=env.penalty.std * 6)


def test_resetting_environment(env):
    env.reset()
    all(_ for _ in until_done(env, go_left()))
    env.reset()
    assert sum(1 for _ in until_done(env, 0)) == env.len_episode


def test_on_average_random_agent_performs_poorly(env, sample_average_reward):
    assert sample_average_reward(env, 100) <= env.reward_range[1] * 0.5


def test_render_writes_current_state_to_stdout(env, make_state_string, capstdout):
    force_reset(env, left)
    env.render()
    assert capstdout.read() == "\n" + make_state_string(agent_pos=0) + "\n"
    env.step(0)
    env.render()
    assert capstdout.read() == "(Right)\n" + make_state_string(agent_pos=1, reward=True) + "\n"
    env.step(0)
    env.render()
    assert capstdout.read() == "(Right)\n" + make_state_string(agent_pos=1, reward=False) + "\n"
    env.step(1)
    env.render()
    assert capstdout.read() == "(Left)\n" + make_state_string(agent_pos=0, reward=True) + "\n"


def test_alternation_example(request, capstdout):
    example = request.session.fspath / "examples/alternation.py"
    lines = run_example(example)
    assert "Observation: " in last(lines)
