[![Build Status](https://travis-ci.org/SwamyDev/gym-quickcheck.svg?branch=master)](https://travis-ci.org/SwamyDev/gym-quickcheck) [![Coverage Status](https://coveralls.io/repos/github/SwamyDev/gym-quickcheck/badge.svg?branch=master)](https://coveralls.io/github/SwamyDev/gym-quickcheck?branch=master)

# gym-quickcheck
Many bugs and implementation errors can already be spotted by running the agent in relatively simple environments. This gym extension provides environments which run fast even on low spec VMs and can be used in Continuous Integration tests. This project aims to help improve code quality and stability of Reinforcement Learning algorithms by providing additional means for automated testing.

## Installation
You can install from source using pip:
```bash
pip install .
```

## Quick Start
A random agent navigating the random walk environment, rendering a textual representation to the standard output:

[embedmd]:# (examples/random_walk.py python)
```python
import gym

env = gym.make('gym_quickcheck:random-walk-v0')
done = False
observation = env.reset()
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    print(f"Observation: {observation}, Reward{reward}")
```

## Random Walk
This random walk environment is similar to the one described in [Reinforcement Learning An Introduction](http://incompleteideas.net/book/the-book-2nd.html). It differs in having max episode length instead of terminating at both ends, and in penalizing each step except the goal.

![random walk graph](assets/random-walk.png)

The agent receives a reward of 1 when it reaches the goal, which is the rightmost cell and -1 on reaching any other cell. The environment either terminates upon reaching the goal or after a maximum amount of steps. First, this ensures that the environment has an upper bound of episodes it takes to complete, making testing faster. Second, because the maximum negative reward has a lower bound that is reached quickly, reasonable baseline estimates should improve learning significantly. With baselines having such a noticeable effect, it makes this environment well suited for testing algorithms which make use of baseline estimates. 
