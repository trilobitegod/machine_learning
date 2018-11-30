# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 05:56:42 2018

@author: Snake
"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(100):
        observation = env.reset()
        while True:
            env.reader()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if done:
                break
    print('game over')
    env.destroy()
    
    
if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(action=list(range(env.n_actions)))
    
    env.after(100, update)
    env.mainloop()