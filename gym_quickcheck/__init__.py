from gym.envs.registration import register


register(
    id='random-walk-v0',
    entry_point='gym_quickcheck.envs:RandomWalkEnv',
)
