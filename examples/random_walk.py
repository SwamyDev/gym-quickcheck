import gym

env = gym.make('gym_quickcheck:random-walk-v0')
done = False
observation = env.reset()
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    print(f"Observation: {observation}, Reward{reward}")
