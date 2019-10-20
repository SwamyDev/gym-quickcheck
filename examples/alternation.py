import gym

env = gym.make('gym_quickcheck:alternation-v0')
done = False
observation = env.reset()
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    print(f"Observation: {observation}, Reward: {reward}")
