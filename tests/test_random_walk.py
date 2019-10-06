import numpy as np
import pytest
from gym import utils
from more_itertools import last

from gym_quickcheck.envs import RandomWalkEnv


@pytest.fixture
def env():
    return RandomWalkEnv()


@pytest.fixture
def obs_shape(env):
    return env.observation_space.shape


@pytest.fixture
def make_observation(obs_shape):
    def obs_fac(agent_pos):
        obs = np.zeros(shape=obs_shape)
        obs[agent_pos] = 1
        return obs

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
def capstdout(capsys):
    class _CapStdOut:
        def __init__(self, cap):
            self._cap = cap

        def read(self):
            return self._cap.readouterr()[0]

    return _CapStdOut(capsys)


@pytest.fixture
def make_walk_string(walk_len):
    def walk_fac(agent_pos, color='red'):
        s = ['#'] * walk_len
        s[agent_pos] = utils.colorize(s[agent_pos], color=color, highlight=True)
        return "".join(s)

    return walk_fac


GYM_INTERFACE = [('reset', ()), ('step', (0,)), ('render', ()), ('close', ())]
GYM_PROPERTIES = ['action_space', 'observation_space']


def test_adherence_to_gym_contract(env):
    list(getattr(env, func)(*args) for func, args in GYM_INTERFACE)
    assert all(getattr(env, p) is not None for p in GYM_PROPERTIES)


def test_agent_starts_in_the_center(env, make_observation, center):
    assert_obs_eq(env.reset(), make_observation(agent_pos=center))


def assert_obs_eq(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def test_navigating_the_agent(env, make_observation, center):
    actions = [1, 1, 0, 0, 0, 1]
    offsets = [1, 2, 1, 0, -1, 0]
    for a, o in zip(actions, offsets):
        assert_obs_eq(unpack_obs(env.step(a)), make_observation(agent_pos=center + o))


def unpack_obs(step_tuple):
    return step_tuple[0]


def test_reset_moves_agent_back_to_center(env, make_observation, center):
    env.step(0)
    assert_obs_eq(env.reset(), make_observation(agent_pos=center))


def test_environment_does_not_finish_until_goal_or_max_length_is_reached(env):
    assert all(not unpack_done(env.step(0)) for _ in range(env.max_len - 1))


def unpack_done(step_tuple):
    return step_tuple[2]


@pytest.mark.parametrize('direction', [0, 1])
def test_reset_environment_is_not_done(env, direction):
    all(until_done(env, direction))
    env.reset()
    assert not unpack_done(env.step(direction))
    all(until_done(env, direction))


def until_done(env, direction):
    done = False
    while not done:
        a = direction if isinstance(direction, int) else direction()
        o, r, done, _ = env.step(a)
        yield o, r, done, _


def test_environment_has_a_max_episode_len(env):
    assert sum(1 for _ in until_done(env, 0)) == env.max_len


def test_finishes_when_reaching_right_most_edge(env, steps_to_edge):
    assert last(unpack_done(env.step(1)) for _ in range(steps_to_edge))


def test_each_step_outside_of_goal_returns_a_penalty(env, walk_len, steps_to_edge):
    assert all(unpack_reward(env.step(0)) == env.penalty for _ in range(steps_to_edge))
    assert all(unpack_reward(env.step(1)) == env.penalty for _ in range(walk_len - 2))


def unpack_reward(step_tuple):
    return step_tuple[1]


def test_reaching_goal_on_the_right_returns_reward(env):
    assert last(r for _, r, _, _ in until_done(env, 1)) == env.reward


def test_walking_right_achieves_maximum_reward(env):
    assert sum(r for _, r, _, _ in until_done(env, 1)) == env.reward_range[1]


def test_walking_left_until_max_length_is_reached_achieves_minimum_reward(env):
    assert sum(r for _, r, _, _ in until_done(env, 0)) == env.reward_range[0]


def test_on_average_random_agent_performs_poorly(env):
    total = 0
    count = 1000
    for _ in range(0, count):
        total += sum(r for _, r, _, _ in until_done(env, env.action_space.sample))
        env.reset()
    assert (total / count) <= np.mean(env.reward_range)


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
