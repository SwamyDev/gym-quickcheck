import numpy as np
import pytest
from gym import utils
from more_itertools import last
from pytest import approx

from gym_quickcheck.envs.n_knob_env import NKnobEnv, ResetError
from tests.aux import assert_that, follows_contract, assert_obs_eq, until_done, unpack_done, unpack_reward, unpack_obs, \
    run_example


@pytest.fixture
def env():
    return NKnobEnv()


@pytest.fixture
def idle(env):
    return make_action(np.zeros_like(env.knobs))


@pytest.fixture
def make_knob_string(env):
    def knob_fac(action):
        s = []
        for k, a in zip(env.knobs, action):
            color = 'green' if abs(k - a) < env.sensitivity else 'red'
            s.append(utils.colorize(f"({k:.3f}/{a:.3f})", color=color, highlight=True))
        return " ".join(s)

    return knob_fac


def make_action(values):
    return np.array(values)


def test_adherence_to_gym_contract(env, gym_interface, gym_properties):
    assert_that(env, follows_contract(gym_interface, gym_properties))


def test_initial_observation_is_direction_from_zero(env):
    env.seed(42)
    assert_obs_eq(env.reset(), make_obs([-1, -0, 1]))
    assert_obs_eq(env.reset(), make_obs([-0, 1, 1]))


def make_obs(values):
    return np.array(values)


def test_raise_error_when_stepping_without_resetting(env, idle):
    with pytest.raises(ResetError):
        env.step(idle)


def test_reward_range_is_dependent_on_length_and_knobs(env):
    env.reset()
    assert env.reward_range == (-env.max_len * 2 * len(env.knobs), 0)


def test_has_a_max_episode_length(env, idle):
    env.reset()
    assert sum(1 for _ in until_done(env, lambda: idle)) == env.max_len


def test_not_done_until_finished_or_reset(env, idle):
    env.reset()
    assert all(not unpack_done(env.step(idle)) for _ in range(env.max_len - 1))
    assert unpack_done(env.step(idle))
    env.reset()
    assert not unpack_done(env.step(idle))


def test_each_step_gives_as_reward_the_distance_to_the_knobs(env):
    env.reset()
    act = np.random.rand(env.knobs.shape[0])
    assert unpack_reward(env.step(act)) == -np.linalg.norm(env.knobs - act)


def test_observation_from_step_indicates_direction_of_solution(env, idle):
    env.seed(42)
    initial = env.reset()
    assert_obs_eq(unpack_obs(env.step(idle)), initial)
    assert_obs_eq(unpack_obs(env.step(initial)), initial * -1)


def test_setting_the_knobs_to_their_correct_value_solves_the_environment(env):
    env.reset()
    obs, reward, done, _ = env.step(env.knobs)
    assert done
    assert_obs_eq(obs, np.zeros_like(obs))
    assert reward == approx(0.0)


def test_solving_by_within_sensitivity(env):
    env.reset()
    epsilon = env.observation_space.sample() * env.sensitivity * 0.999
    obs, r, done, _ = env.step(env.knobs + epsilon)
    assert done


def test_on_average_random_agent_performs_poorly(env, sample_average_reward):
    assert sample_average_reward(env, 100) <= -env.max_len * 0.9


def test_render_writes_current_state_to_stdout(env, make_knob_string, capstdout):
    env.reset()
    env.render()
    assert capstdout.read() == make_knob_string(make_action(np.zeros_like(env.knobs))) + "\n"
    env.step(make_action(np.ones_like(env.knobs)))
    env.render()
    assert capstdout.read() == make_knob_string(make_action(np.ones_like(env.knobs))) + "\n"
    env.step(env.knobs)
    env.render()
    assert capstdout.read() == make_knob_string(env.knobs) + "\n"


def test_n_knob_example(request, capstdout):
    example = request.session.fspath / "examples/n_knob.py"
    lines = run_example(example)
    assert "Observation: " in last(lines)
