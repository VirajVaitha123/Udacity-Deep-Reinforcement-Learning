import numpy as np
from collections import defaultdict
import random


class Agent:

    def __init__(self, nA=6, eps = 1, gamma = 1.0, alpha = 0.01):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - Q: Q-table stores action values
        - eps: epsilon value which will decay over time allowing our agent to initially explore different stratigies and eventually start to exploit the best  action (Epsilon -greedy
        - gamma: discount rate"""
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = eps/i_episode
        self.gamma = gamma
        self.gamma = alpha

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.eps/i_episode:
            return np.argmax(self.Q[state])
        else:                     # otherwise, select an action randomly
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            next_action = select_action(state)
            current = self.Q[state][action]
            Qsa_next = self.Q[next_state][next_action] if next_state is not None else 0
            target = reward + (self.gamma * Qsa_next)               # construct TD target
            new_value = current + (self.alpha * (target - current)) # get updated value
            self.Q[state][action] = new_value
       
        if done:
            next_state=None
            next_action=None
            current = self.Q[state][action]
            Qsa_next = self.Q[next_state][next_action] if next_state is not None else 0
            target = reward + (self.gamma * Qsa_next)               # construct TD target
            new_value = current + (self.alpha * (target - current)) # get updated value
            self.Q[state][action] = new_value