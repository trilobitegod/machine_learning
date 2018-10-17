# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 00:28:12 2018

@author: Snake
"""

import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSMT,TimeDistributed,Dense
from keras.optimizers import Adam

BATCH_INDEX = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 1
