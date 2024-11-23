---
title: "Solving the Taxi Problem using Reinforcement Learning techniques."
excerpt: "Applying SARSA, Q-Learning and off-policy monte carlo method to solve the Taxi Problem.>"
collection: portfolio
---

### The Taxi Problem
<hr style="border:2px solid gray">

The Taxi problem can be described as follows:
```python
        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+
```
There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends. The rewards are:
	•	-1 per step unless other reward is triggered.
	•	+20 delivering passenger.
	•	-10 executing "pickup" and "drop-off" actions illegally.
If a navigation action would cause the taxi to hit a wall (solid borders in the map), the action is a no-op, and there is only the usual reward of −1.


```python
import numpy as np
from collections import defaultdict
```

Let us import the TaxiEnv class from env_taxi.py and create an instance of the Taxi environment.

```python
from env_taxi import TaxiEnv
env=TaxiEnv()
```

Then, we reset the environment:

```python
np.random.seed(100)
env.reset()
```
(2, 3, 2, 1)


It returns four state variables:
	•	Row index of the taxi (starting from 0 in Python)
	•	Column index of the taxi (starting from 0 in Python)
	•	Passenger location (0-4): 0=R, 1=G, 2=Y, 3=B, 4=in taxi
	•	Destination location (0-3): same encoding rule as that for passenger location but excluding the "in taxi" option.

We can use the describe_state method of the Taxi instance to display the state variables without location encoding.

```python
env.describe_state()
```
{'Taxi Location': [2, 3], 'Passenger Location': 'Y', 'Destination Index': 'G'}


We can use the render method to visualize the state.

```python
env.render()
```
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+


There are 6 discrete deterministic actions:
	•	'South': move south
	•	'West': move north
	•	'East': move east
	•	'West': move west
	•	'Pickup': pickup passenger
	•	'Dropoff': drop off passenger


```python
print(env.action_space)
['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
env.locs
```
[(0, 0), (0, 4), (4, 0), (4, 3)]


Let us move one step to west:

```python
env.step('West')
```
((2, 2, 2, 1), -1, False)


The output is a 3-tuple: the new state (a list of 4 numbers), reward, and whether the episode is ended.

Let us visualize the new state. Note that the last action is shown at the bottom as well.

```python
env.render()
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
  (West)
```

We begin by generating one episode for the taxi driver who:
	•	Picks up the passenger when the taxi is at the location of the passenger when they are not yet at the destination;
	•	Drops off the passenger in the taxi when reaching the destination;
	•	Moves randomly with equal probabilities when finding the passenger or when the passenger is in the taxi (but not yet arriving at destination).


First initialize the policy as a dictionary pi_naive . Assigning the actions 'Pickup' and 'Dropoff' to the corresponding states.

```python
pi_naive= defaultdict(lambda: np.random.choice(env.action_space))

for pass_idx, loc in enumerate(env.locs):
    for dest_idx in range(len(env.locs)):
        if pass_idx != dest_idx:  # Passenger and destination are different
            # At the passenger's location and not in the taxi yet
            pi_naive[(loc[0], loc[1], pass_idx, dest_idx)] = 'Pickup'
            # At the destination with the passenger in the taxi
            pi_naive[(env.locs[dest_idx][0], env.locs[dest_idx][1], 4, dest_idx)] = 'Dropoff'

```

Reset the environment.

```python
state=env.reset(271)
env.render()
print(env.describe_state())
```
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+

{'Taxi Location': [2, 3], 'Passenger Location': 'Y', 'Destination Index': 'B'}


Now, it’s time to import the simulate_episode function from env_taxi.py to help you simulate episode(s). We don't need to worry about coding the random moves: this function will generate a random move if the state is not provided in the policy dictionary.

```python
from env_taxi import simulate_episode
help(simulate_episode)
Help on function simulate_episode in module env_taxi:

simulate_episode(env, policy, max_iter=inf)
    Simulate a episode following a given policy
    @param env: game environment
    @param policy: policy (a dictionary or a function)
    @return: resulting states, actions and rewards for the entire episode
```

We use this function to generate an episode and print the return for this episode.

```python
np.random.seed(13322842)

# Initialize the return estimate
G=0
# Save the sum of rewards in this variable G
states, actions, rewards = simulate_episode(env, pi_naive)
G=sum(rewards)

# Print the return for this episode
print("Return for this episode:", G)
Return for this episode: -108

```

Monte Carlo methods with Exploring Starts is infeasible for solving the optimal policy in the Taxi problem due to the large state-action space, which consists of 500 states (No. Taxi positions x No. Passenger locations x No. Destination locations) and 6 possible actions per state, resulting in 3000 state-action pairs. Ensuring that every state-action pair is explored sufficiently is challenging and computationally intensive. Furthermore it is required that at the beginning of an episode, every state-action pair has a non zero probability of being selected. However in the taxi problem, certain state-action pairs are inaccessible at the beginning of an episode due to the nature of the problem (e.g taxi cant start from locations outside grid boundaries). Temporal difference learning techniques such as Q-Learning and SARSA are better suited as they can better handle large state spaces and can better make incremental updates.

Let’s try finding the optimal policy by using Q-learning with 50000 episodes and the exploration probability $\varepsilon=0.1$. Trying two different values of the step-size parameter $\alpha=0.4$ and $\alpha=0.1$. Compare their performance in 10000 new episodes and comment on the similarities or/and differences.


```python

np.random.seed(13322842)

# Save the policy for alpha=0.4 as a Python dictionary/defaultdic object pi_qlearning_1
pi_qlearning_1= defaultdict(lambda: np.random.choice(env.action_space))

# Save the policy for alpha=0.1 as a Python dictionary/defaultdic object pi_qlearning_2
pi_qlearning_2= defaultdict(lambda: np.random.choice(env.action_space))

def gen_epsilon_greedy_policy(action_space, Q, epsilon):

    def epsilon_greedy_policy(state):
        greedy_action=max(Q[state],key=Q[state].get)
        random_action=np.random.choice(action_space)
        return np.random.choice([random_action,greedy_action],p=[epsilon,1-epsilon])

    return epsilon_greedy_policy


def q_learning(env, n_episode, alpha, gamma = 1, epsilon=0.1):
    length_episode = np.zeros(n_episode) 
    sum_reward_episode = np.zeros(n_episode)
    
    Q = defaultdict(lambda: dict(zip(env.action_space, np.zeros(len(env.action_space)))))
    epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space, Q, epsilon)
    
    pi= defaultdict(lambda: np.random.choice(env.action_space))
    
    for episode in range(n_episode):
        
        state = env.reset()
        is_done = False
        
        action = epsilon_greedy_policy(state)
        while not is_done:
            next_state, reward, is_done = env.step(action)
            next_action = epsilon_greedy_policy(next_state)
            
            v= max(Q[next_state].values())
            td_delta = reward + gamma * v - Q[state][action]
            
            Q[state][action] += alpha * td_delta
            pi[state]=max(Q[state],key=Q[state].get)
            
            length_episode[episode] += 1
            sum_reward_episode[episode] += reward
            
            if is_done:
                break
            state = next_state
            action = next_action
              
    return pi

num_episodes = 50000

# Train Q-learning policies with different alpha values
pi_qlearning_1 = q_learning(env, num_episodes, alpha=0.4)
pi_qlearning_2 = q_learning(env, num_episodes, alpha=0.1)  

# Merge them with the naive policy: make sure to pick up and drop off passengers correctly.
pi_qlearning_1.update(pi_naive)
pi_qlearning_2.update(pi_naive)

```


# Compare their performance in 10000 new episodes
from env_taxi import performance_evaluation
print('--------  Q-learning, alpha=0.4  --------')
performance_evaluation(env,pi_qlearning_1)
print('--------  Q-learning, alpha=0.1  --------')
performance_evaluation(env,pi_qlearning_2)
--------  Q-learning, alpha=0.4  --------
100%|██████████| 10000/10000 [00:04<00:00, 2131.13it/s]
The sum of rewards per episode: 7.8886
The percentage of episodes that cannot terminate within 1000 steps: 0.0
--------  Q-learning, alpha=0.1  --------
100%|██████████| 10000/10000 [00:04<00:00, 2163.29it/s]
The sum of rewards per episode: 7.8886
The percentage of episodes that cannot terminate within 1000 steps: 0.0


We would expect Q-Learning to perform well on the Taxi Problem given that it is equipped to handle larger state-action spaces via incremental updates to the action-value function. More so, it uses bootstrapping and updating based on the current estimates, meaning it can learn from partial episodes and improve action-value estimates progressively. This is beneficial for the taxi problem with its long episodes and sparse rewards. We observe that for both learning rates, the performance is the same and good. A higher learning rate means the algorithm adapts faster to new information, whereas a lower rate is more stable. In our case, performance is identical, showing that there is little sensitivity to the learning rate in this problem when using Q-Learning.

We now try to find the optimal $\varepsilon$-soft policy by using Sarsa with 50000 episodes and the exploration probability $\varepsilon=0.1$. Try two different values of the step-size parameter $\alpha=0.4$ and $\alpha=0.1$. Compare their performance, and comment on the similarties or/and differences with that of Q-learning.
First, let us find out the greedy action chosen by Sarsa.

```python
np.random.seed(1)

# Save the policy for alpha=0.4 as a Python dictionary/defaultdic object pi_sarsa_1
pi_sarsa_1_greedy= defaultdict(lambda: np.random.choice(env.action_space))

# Save the policy for alpha=0.1 as a Python dictionary/defaultdic object pi_sarsa_2
pi_sarsa_2_greedy= defaultdict(lambda: np.random.choice(env.action_space))


def sarsa(env, n_episode, alpha, gamma = 1, epsilon = 0.1):
    
    length_episode = np.zeros(n_episode) 
    sum_reward_episode = np.zeros(n_episode)
    
    Q = defaultdict(lambda: dict(zip(env.action_space, np.zeros(len(env.action_space)))))
    
    pi = gen_epsilon_greedy_policy(env.action_space, Q, epsilon)
    
    for episode in range(n_episode):
        
        state = env.reset()
        is_done = False
        action = pi(state)
        while not is_done:
            next_state, reward, is_done = env.step(action)
            next_action = pi(next_state)
            td_delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * td_delta
            
            length_episode[episode] += 1
            sum_reward_episode[episode] += reward
            if is_done:
                break
            state = next_state
            action = next_action
            
            
    return Q

num_episodes = 50000

#First we run SARA to retrieve the dictionary Q
Q1 = sarsa(env, num_episodes, alpha = 0.4)
Q2 = sarsa(env, num_episodes, alpha = 0.1)

#Now we extract the greedy actions and save them in the dictionary
for key, actions in Q1.items():
    best_action = max(actions, key=actions.get)
    pi_sarsa_1_greedy[key] = best_action

for key, actions in Q2.items():
    best_action = max(actions, key=actions.get)
    pi_sarsa_2_greedy[key] = best_action

# End Coding Here

# Merge them with the naive policy: make sure to pick up and drop off passengers correctly.

pi_sarsa_1_greedy.update(pi_naive)
pi_sarsa_2_greedy.update(pi_naive)
```

Now generate the $\varepsilon$-greedy policies with $\varepsilon=0.1$ for different $\alpha=0.4,0.1$. Then evaluate the performance of these two $\varepsilon$-greedy policies.

```python
# The epsilon-greedy policies shall be stored as Python functions.

# Start Coding Here

# Save the policy for alpha=0.4 as a Python dictionary/defaultdic object pi_sarsa_1
pi_sarsa_1= defaultdict(lambda: np.random.choice(env.action_space))

# Save the policy for alpha=0.1 as a Python dictionary/defaultdic object pi_sarsa_2
pi_sarsa_2= defaultdict(lambda: np.random.choice(env.action_space))

def sarsa(env, n_episode, alpha, gamma = 0.99, epsilon = 0.1):
    length_episode = np.zeros(n_episode) 
    sum_reward_episode = np.zeros(n_episode)
    
    Q = defaultdict(lambda: dict(zip(env.action_space, np.zeros(len(env.action_space)))))
    
    pi = gen_epsilon_greedy_policy(env.action_space, Q, epsilon)
    
    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        action = pi(state)
        while not is_done:
            next_state, reward, is_done = env.step(action)
            next_action = pi(next_state)
            td_delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * td_delta
            
            length_episode[episode] += 1
            sum_reward_episode[episode] += 1
            
            if is_done:
                break
            state = next_state
            action = next_action
            
    return(pi)

pi_sarsa_1 = sarsa(env, num_episodes, alpha=0.4)
pi_sarsa_2 = sarsa(env, num_episodes, alpha=0.1)
            
from env_taxi import performance_evaluation
print('--------  Sarsa, alpha=0.4  --------')
performance_evaluation(env,pi_sarsa_1)
print('--------  Sarsa, alpha=0.1  --------')
performance_evaluation(env,pi_sarsa_2)

```

--------  Sarsa, alpha=0.4  --------
100%|██████████| 10000/10000 [00:19<00:00, 525.36it/s]
The sum of rewards per episode: -1.5143
The percentage of episodes that cannot terminate within 1000 steps: 0.0
--------  Sarsa, alpha=0.1  --------
100%|██████████| 10000/10000 [00:14<00:00, 677.41it/s]
The sum of rewards per episode: 2.3632
The percentage of episodes that cannot terminate within 1000 steps: 0.0


SARSA is similar to Q-Learning, but differs in that it is an on-policy algorithm that updates the action-value function using the reward of the next state-action pair according to the current policy. Q-Learning on the other hand is an off-policy algorithm that updates using the maximum possible reward of the next state, independent of the policy followed. This makes Q-Learning more aggressive as it focuses on the best possible action in each state. SARSA is more conservative because it updates the action-value function based on the actual actions taken, which include exploratory moves. This may explain why the sum of rewards per episode for SARSA is lower in this case. The more aggressive exploitation of rewards of Q-Learning makes it yield a higher sum of rewards per episode. We also observe that the sum of rewards per episode were higher for alpha = 0.1. This indicates that the SARSA algorithm is more sensitive to changes in the learning rate for the taxi problem. Similar arguments hold here as for Q-Learning. The lower learning rate is better in this scenario as it results in more stable and incremental updates to the action-value function. This makes it better suited for the large state-action space and sparse rewards. Whereas Q-Learning may get away with a larger learning rate due to updating according to the maximum reward of the next state, SARSA does not do this such that a higher learning rate has a more negative impact on the sum of rewards than it did for Q-Learning.

Double Q-Learning is used to address the overestimation bias of standard Q-Learning, whereby action values are overestimated because the same value function is used for both selecting and evaluating actions. Double Q-Learning uses two separate value functions (Q1 and Q2), one for action selection and one for action evaluation. This should improve value estimates and give better performance as a result. Given that the Taxi problem is not particularly complex and the convergence time of the Q-Learning algorithm is already relatively low, it is unlikely to yield any substantial improvement. The state-action space is small enough for Q-Learning and there is a straight forward reward structure, which contribute to a low risk of overestimation bias. This explains why Q-Learning already performs well, indicating that double Q-Learning would yield limited improvement (if any).

Now we investigate whether we can solve the optimal policy by using off-policy Monte Carlo control with weighted importance sampling. Use the optimal $\varepsilon$-soft policy found by Sarsa above with $\alpha=0.1$ as the behavior policy.

```python
np.random.seed(13322842)

# Initialize policy as a dictionary object
pi_wis= defaultdict(lambda: np.random.choice(env.action_space))


def mc(env, num_episodes, behaviour_policy):
    pi = defaultdict(lambda: np.random.choice(env.action_space))
    Q = defaultdict(lambda: dict(zip(env.action_space, np.zeros(len(env.action_space)))))
    C = defaultdict(lambda: dict(zip(env.action_space, np.zeros(len(env.action_space)))))
    
    for _ in range(num_episodes):
        states, actions, rewards = simulate_episode(env, behaviour_policy)
        G = 0
        W = 1
        
        for state, action, reward in zip(states[::-1][1:], actions[::-1], rewards[::-1]):
            G = G + reward
            C[state][action] = W + C[state][action]
            Q[state][action] = (W / C[state][action]) * (G - Q[state][action]) + Q[state][action]
            pi[state] = max(Q[state], key = Q[state].get)
            
            if action != pi[state]:
                break
            
            W = W / (1/len(env.action_space))
    
    return pi

pi_wis = mc(env, num_episodes, pi_sarsa_2)

pi_wis.update(pi_naive)

```

Let us evaluate its performance:
performance_evaluation(env,pi_wis)
100%|██████████| 10000/10000 [00:05<00:00, 1891.62it/s]
The sum of rewards per episode: 7.3223
The percentage of episodes that cannot terminate within 1000 steps: 0.0


The conclusion is that Q-Learning is best suited for the taxi problem, given that it yielded the highest sum of rewards per episode and converged quickly. This can likely be attributed to the more aggressive approach of updating using the maximum possible reward of the next step, irrespective of the policy followed. A possible issue with Q-Learning may be the overestimation bias, which leads to less accurate value estimates. However this was fortunately unlikely to be the case in this scenario. Off-policy Monte Carlo control with weighted importance sampling also performed well. This is likely due to its ability to be data efficient and more accurate due to unbiased value estimates. It also performed well for the taxi problem given the structured environment of the problem overall. A disadvantage is its complexity relative to the other two methods as it needs to manage the importance sampling ratios. SARSA yielded the comparatively worst performance, likely due to its overly conservative approach in updating. Generally speaking, the conservative updating is beneficial as it leads to more stable learning and convergence. However in this case, it was triumphed by the aggression of Q-Learning and data efficiency of MC control with weighted importance sampling.

