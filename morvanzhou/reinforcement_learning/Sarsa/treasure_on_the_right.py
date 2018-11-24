# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 01:08:30 2018

@author: Snake
"""

import numpy as np
import panda as pdd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['Left', 'Right']
EPSILON = 0.9
ALPHA = 0.2
GAMMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3

def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, length(actions))), column=actions,)
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform() > EPSILON) or ((stata_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S, A):
    if A == 'Right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    if A == 'Left':
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S-1
    return S_, R


                