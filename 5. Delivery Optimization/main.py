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


env = DeliveryEnvironment(n_stops = 1967,method = "plan")
agent = DeliveryQAgent(env.observation_space,env.action_space)

#e,a,env_min = run_n_episodes(env,agent,"training_100_stops.gif")

run_n_episodes(env,agent,"training_1967_stops_dist.gif",n_episodes = 1)

env.render()
env.extract_csv("test.csv")
#env_min.render()