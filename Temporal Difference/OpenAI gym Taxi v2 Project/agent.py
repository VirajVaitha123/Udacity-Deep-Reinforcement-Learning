import numpy as np
from collections import defaultdict
import random


class Agent:

    def __init__(self, nA=6, eps = 0.0001, gamma = 1.0, alpha = 0.01, eps_decay = 0.9999):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - Q: Q-table stores action values
        - eps: epsilon value which will decay over time allowing our agent to initially explore different stratigies and eventually start to exploit the best  action (Epsilon -greedy
        - gamma: discount rate"""
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = eps                                      ###!!!first mention of eps (and in the top line arguments line 8) 
        self.gamma = gamma
        self.alpha = alpha

    def select_action(self, state):

        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.eps/i_episode:             ###!!!This is the main issue, obviously this won't work here :(
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
            next_action = self.select_action(state)             ###!!!If not done,I want the next action. Is it ok practice to reference the select_action function like this?
            current = self.Q[state][action]
            Qsa_next = self.Q[next_state][next_action] if next_state is not None else 0
            target = reward + (self.gamma * Qsa_next)               # construct TD target
            new_value = current + (self.alpha * (target - current)) # get updated value
            self.Q[state][action] = new_value
       
        if done:                                                ###!!!I don't see why we need the if done statement as we have the if next_state is not None else 0 line. Is it to apply the break?
            next_state=None
            next_action=None
            current = self.Q[state][action]
            Qsa_next = self.Q[next_state][next_action] if next_state is not None else 0
            target = reward + (self.gamma * Qsa_next)               # construct TD target
            new_value = current + (self.alpha * (target - current)) # get updated value
            self.Q[state][action] = new_value
            break                                              ###!!! This break causes a problem, I think I can run without it as it's in the monitor.py?
            