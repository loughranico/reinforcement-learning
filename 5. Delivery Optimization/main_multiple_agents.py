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



env = DeliveryEnvironment(n_stops = 1967,method = "plan")

agents = []

truck_file ="./data/camion.csv"
with open(truck_file) as f:
    r = csv.reader(f)
    trucks = defaultdict(list)
    for nombre,lon,lat in r:
        if nombre != "idCamion":
            agents.append(DeliveryQAgent(env.observation_space,env.action_space,name = nombre,x_base = lon,y_base = lat))

#e,a,env_min = run_n_episodes(env,agent,"training_100_stops.gif")

run_n_episodes_ma(env,agents,"training_1967_stops_plan_multiagents.gif",n_episodes = 1)

env.render()
#env_min.render()