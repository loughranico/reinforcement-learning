# Base Data Science snippet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm_notebook
import sys
sys.path.append("../")

from delivery import *

'''
#env = DeliveryEnvironment(n_stops = 50)
#env = DeliveryEnvironment(n_stops = 2000, max_box = 1000)




env.render()

print(f"The first stop id: {env.stops}")

for i in range(4):
    env.step(i)
    print(f"Stops visited in step {i}: {env.stops}")

env.render()'''


env = DeliveryEnvironment(n_stops = 10,method = "distance")
agent = DeliveryQAgent(env.observation_space,env.action_space)

#e,a,env_min = run_n_episodes(env,agent,"training_100_stops.gif")

run_n_episodes(env,agent,"training_10_stops.gif",n_episodes = 2000)

env.render()
#env_min.render()