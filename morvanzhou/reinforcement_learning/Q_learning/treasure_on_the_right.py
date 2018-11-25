# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 01:08:30 2018

@author: Snake
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)   # reproducible

N_STATES = 6    # the length of 1-dimensional world
ACTIONS = ['Left', 'Right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.2     # learning rate
GAMMA = 0.9     # discounter factor
MAX_EPISODES = 13   # max episodes
FRESH_TIME = 0.1    # fresh time for one move

def build_q_table(n_states, actions):
    # q_table initial values
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions,)
    return table

def choose_action(state, q_table):
    # how to choose an action
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S, A):
    # how agent interact with environment
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

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
        
def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            update_env(S, episode, step_counter)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
        
                
                