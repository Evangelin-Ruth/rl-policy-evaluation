# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.


## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.
## States

The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.
Five Transition states / Non-terminal States including S: The starting state.
## Actions
The agent can take two actions:

R: Move right.
L: Move left.
## Transition Probabilities
The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.
## Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.
## Graphical Representation

![image](https://github.com/Evangelin-Ruth/rl-policy-evaluation/assets/94219798/a2b77997-7f5d-4140-82d4-e87acc010d98)

## POLICY EVALUATION FUNCTION
![image](https://github.com/Evangelin-Ruth/rl-policy-evaluation/assets/94219798/678cbb13-04a1-4d6b-b0af-fcc7bfdaaf2c)

## PROGRAM
```
pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk

import warnings ; warnings.filterwarnings('ignore')
import gym, gym_walk
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)
```
## OUTPUT:
Mention the first and second policies along with its state value function and compare them

## RESULT:

Write your result here
