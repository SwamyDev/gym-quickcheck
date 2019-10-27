[![Build Status](https://travis-ci.org/SwamyDev/gym-quickcheck.svg?branch=master)](https://travis-ci.org/SwamyDev/gym-quickcheck) [![Coverage Status](https://coveralls.io/repos/github/SwamyDev/gym-quickcheck/badge.svg?branch=master)](https://coveralls.io/github/SwamyDev/gym-quickcheck?branch=master) [![PyPI version](https://badge.fury.io/py/gym-quickcheck.svg)](https://badge.fury.io/py/gym-quickcheck)

# gym-quickcheck
Many bugs and implementation errors can already be spotted by running the agent in relatively simple environments. This gym extension provides environments which run fast even on low spec VMs and can be used in Continuous Integration tests. This project aims to help improve code quality and stability of Reinforcement Learning algorithms by providing additional means for automated testing.

## Installation
You can install the package using pip:
```bash
pip install gym-quickcheck
```

## Quick Start

### Random Walk
A random agent navigating the random walk environment, rendering a textual representation to the standard output:

[embedmd]:# (examples/random_walk.py python)
```python
import gym

env = gym.make('gym_quickcheck:random-walk-v0')
done = False
observation = env.reset()
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    print(f"Observation: {observation}, Reward: {reward}")
```

Running the example should produce an output similar to this:
```
...
(Left)
#######
Observation: [0. 0. 0. 0. 0. 1. 0.], Reward: -1
(Right)
#######
Observation: [0. 0. 0. 0. 0. 0. 1.], Reward: 1
```

### Alternation
A random agent navigating the alteration environment, rendering a textual representation to the standard output:

[embedmd]:# (examples/alternation.py python)
```python
import gym

env = gym.make('gym_quickcheck:alternation-v0')
done = False
observation = env.reset()
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    print(f"Observation: {observation}, Reward: {reward}")
```

Running the example should produce an output similar to this:
```
...
(Right)
##
Observation: [0 1], Reward: -0.9959229664071392
(Left)
##
Observation: [1 0], Reward: 0.8693727604523271
```

## Random Walk
This random walk environment is similar to the one described in [Reinforcement Learning An Introduction](http://incompleteideas.net/book/the-book-2nd.html). It differs in having max episode length instead of terminating at both ends, and in penalizing each step except the goal.

![random walk graph](assets/random-walk.png)

The agent receives a reward of 1 when it reaches the goal, which is the rightmost cell and -1 on reaching any other cell. The environment either terminates upon reaching the goal or after a maximum amount of steps. First, this ensures that the environment has an upper bound of episodes it takes to complete, making testing faster. Second, because the maximum negative reward has a lower bound that is reached quickly, reasonable baseline estimates should improve learning significantly. With baselines having such a noticeable effect, it makes this environment well suited for testing algorithms which make use of baseline estimates. 

## Alternation
The alteration environment is straightforward, as it just requires the agent to alternate between its two possible states to achieve the maximum reward.

![alteration graph](assets/alteration.png)

The agent receives a normally distributed reward of 1 when switching from one state to the other, and a normally distributes penalty of -1 when staying in its current state. The environment terminates after a fixed amount of steps. This environment's rewards nicely scale linearly with performance. Meaning if the agent alternates one sequence more, it gets precisely one more reward. It makes it easier for agents not to get stuck at local minima. Hence most agents should be able to learn the optimal policy quickly. However, a random agent only achieves, on average, a total reward around zero. It makes this environment well suited for sanity checking algorithms making sure that they learn at all. By providing such a simple setup, it is also easier to comprehend any obvious problems an algorithm might have.  
