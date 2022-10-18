#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING

Started on the 25/08/2017


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys
import random
import time
import random
import numpy as np




from rl import utils
from rl.memory import Memory
from rl.agents.base_agent import Agent



class QAgent(Agent):
    def __init__(self,states_size,actions_size,piece_size,epsilon = 1.0,epsilon_min = 0.01,epsilon_decay = 0.999,gamma = 0.95,lr = 0.8):
        self.states_size = states_size
        self.actions_size = actions_size
        self.piece_size = piece_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q = self.build_model(states_size+1,actions_size+1,piece_size)


    def build_model(self,states_size,actions_size,piece_size):
        Q = np.zeros([states_size,actions_size,piece_size])
        Q[:,states_size:] = -np.inf
        return Q


    def train(self,s,a,r,s_next):
        self.Q[s,a[0],a[1]] = self.Q[s,a[0],a[1]] + self.lr * (r + self.gamma*np.max(self.Q[s_next,a[0],a[1]]) - self.Q[s,a[0],a[1]])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def act(self,s):

        q = self.Q[s,:]

        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            a = np.random.randint(self.actions_size)

        return a


