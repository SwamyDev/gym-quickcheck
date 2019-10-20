import numpy as np
import pytest
from gym import utils
from more_itertools import last

from gym_quickcheck.envs import RandomWalkEnv
from tests.aux import assert_that, follows_contract, assert_obs_eq, unpack_reward, unpack_obs, unpack_done, until_done, \
    run_example


@pytest.fixture
def env():
    return RandomWalkEnv()


@pytest.fixture
def obs_shape(env):
    return env.observation_space.shape


@pytest.fixture
def make_observation(obs_shape, make_observation_of):
    def obs_fac(agent_pos):
        return make_observation_of(obs_shape, agent_pos)

    return obs_fac


@pytest.fixture
def walk_len(obs_shape):
    return obs_shape[0]


@pytest.fixture
def center(walk_len):
    return walk_len // 2


@pytest.fixture
def steps_to_edge(walk_len, center):
    return walk_len - center - 1


@pytest.fixture
def make_walk_string(walk_len):
    def walk_fac(agent_pos, color='red'):
        s = ['#'] * walk_len
        s[agent_pos] = utils.colorize(s[agent_pos], color=color, highlight=True)
        return "".join(s)

    return walk_fac


def test_adherence_to_gym_contract(env, gym_interface, gym_properties):
    assert_that(env, follows_contract(gym_interface, gym_properties))


def test_agent_starts_in_the_center(env, make_observation, center):
    assert_obs_eq(env.reset(), make_observation(agent_pos=center))


def test_navigating_the_agent(env, make_observation, center):
    actions = [1, 1, 0, 0, 0, 1]
    offsets = [1, 2, 1, 0, -1, 0]
    for a, o in zip(actions, offsets):
        assert_obs_eq(unpack_obs(env.step(a)), make_observation(agent_pos=center + o))


def test_reset_moves_agent_back_to_center(env, make_observation, center):
    env.step(0)
    assert_obs_eq(env.reset(), make_observation(agent_pos=center))


def test_environment_does_not_finish_until_goal_or_max_length_is_reached(env):
    assert all(not unpack_done(env.step(0)) for _ in range(env.max_len - 1))


@pytest.mark.parametrize('direction', [0, 1])
def test_reset_environment_is_not_done(env, direction):
    all(until_done(env, direction))
    env.reset()
    assert not unpack_done(env.step(direction))
    all(until_done(env, direction))


def test_environment_has_a_max_episode_len(env):
    assert sum(1 for _ in until_done(env, 0)) == env.max_len


def test_finishes_when_reaching_right_most_edge(env, steps_to_edge):
    assert last(unpack_done(env.step(1)) for _ in range(steps_to_edge))


def test_each_step_outside_of_goal_returns_a_penalty(env, walk_len, steps_to_edge):
    assert all(unpack_reward(env.step(0)) == env.penalty for _ in range(steps_to_edge))
    assert all(unpack_reward(env.step(1)) == env.penalty for _ in range(walk_len - 2))


def test_reaching_goal_on_the_right_returns_reward(env):
    assert last(r for _, r, _, _ in until_done(env, 1)) == env.reward


def test_walking_right_achieves_maximum_reward(env):
    assert sum(r for _, r, _, _ in until_done(env, 1)) == env.reward_range[1]


def test_walking_left_until_max_length_is_reached_achieves_minimum_reward(env):
    assert sum(r for _, r, _, _ in until_done(env, 0)) == env.reward_range[0]


def test_on_average_random_agent_performs_poorly(env, sample_average_reward):
    assert sample_average_reward(env, 1000) <= np.mean(env.reward_range)


def test_render_writes_current_state_to_stdout(env, make_walk_string, center, capstdout):
    env.render()
    assert capstdout.read() == "\n" + make_walk_string(agent_pos=center) + "\n"
    env.step(0)
    env.render()
    assert capstdout.read() == "(Left)\n" + make_walk_string(agent_pos=center - 1) + "\n"
    env.step(1)
    env.render()
    assert capstdout.read() == "(Right)\n" + make_walk_string(agent_pos=center) + "\n"


def test_render_agent_pos_in_green_when_reaching_goal(env, make_walk_string, walk_len, capstdout):
    all(_ for _ in until_done(env, 1))
    env.render()
    assert capstdout.read() == "(Right)\n" + make_walk_string(agent_pos=walk_len - 1, color='green') + "\n"


def test_random_walk_example(request, capstdout):
    example = request.session.fspath / "examples/random_walk.py"
    lines = run_example(example)
    assert "Observation: " in last(lines)
