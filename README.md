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
# Reference https://github.com/mimoralea/gym-walk
```

```
def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
```
```
def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
```
```
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)
```
```
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)
```
## Slippery Walk Five MDP:
```
env = gym.make('SlipperyWalkFive-v0')
P = env.env.P
init_state = env.reset()
goal_state = 6
LEFT, RIGHT = range(2)
P
init_state

state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))

## First Policy
pi_1 = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
print_policy(pi_1, P, action_symbols=('<', '>'), n_cols=7)
## Find the probability of success and the mean return of the first policy
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_1, goal_state=goal_state)*100,
    mean_return(env, pi_1)))

## Create your own policy
pi_2 = lambda s: {
    0:LEFT, 1:RIGHT, 2:RIGHT, 3:LEFT, 4:LEFT, 5:RIGHT, 6:LEFT
}[s]
print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)
## Find the probability of success and the mean return of you your policy
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state)*100,
    mean_return(env, pi_2)))

## Compare your policy with the first policy

### The implementation of first code has resulted in success rate of 3% while the second policy has resulted in improving the result of reaching the goal. The success rate for second policy is 39%.
```
## Policy Evaluation:
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V = np.zeros(len(P))
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s] += prob * (reward + gamma *  prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V - V)) < theta:
        break
      prev_V = V.copy()
    return V

## Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)

## Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

## Comparing policies based on state value function
### The state value function of the second policy V2 is greater than that of the first policy V1, so we conclude that the second policy is the best policy.

V1
print_state_value_function(V1, P, n_cols=7, prec=5)
V2
print_state_value_function(V2, P, n_cols=7, prec=5)
V1>=V2
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```
## OUTPUT:

### Policy 1:
![image](https://github.com/Evangelin-Ruth/rl-policy-evaluation/assets/94219798/05a2950f-d12d-40fd-988c-024f34b7e542)
![image](https://github.com/Evangelin-Ruth/rl-policy-evaluation/assets/94219798/a53b7fee-7187-43a6-9b5a-7c7b7802869e)
![image](https://github.com/Evangelin-Ruth/rl-policy-evaluation/assets/94219798/46959142-c8ea-497a-a8dd-9be38f406107)



### Policy 2:
![image](https://github.com/Evangelin-Ruth/rl-policy-evaluation/assets/94219798/68c31d53-4f83-4159-8853-864469d0d518)
![image](https://github.com/Evangelin-Ruth/rl-policy-evaluation/assets/94219798/d624d8cc-cbf2-4d97-9c0a-944b279669ca)
![image](https://github.com/Evangelin-Ruth/rl-policy-evaluation/assets/94219798/63ea1342-b39c-4a84-9972-eaa8ceff31ec)


### Comparison:
![image](https://github.com/Evangelin-Ruth/rl-policy-evaluation/assets/94219798/f286b4aa-2287-4b3c-be92-c12feeccdc8a)

### Conclusion:
![image](https://github.com/Evangelin-Ruth/rl-policy-evaluation/assets/94219798/e656dd94-98dc-4602-b03d-0e3375f44617)

## RESULT:

Thus, a Python program is developed to evaluate the given policy.

