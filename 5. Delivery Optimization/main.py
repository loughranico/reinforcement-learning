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

import cProfile
import pstats



#env_min.render()

def main(args):
    
    # ### Profiling ###
    # profile = cProfile.Profile()
    # profile.enable()
    # #################
    

    num_stops = 1967
    num_trucks = 107
    iters = int(args[0])
    
    env = DeliveryEnvironment(n_stops = num_stops,n_trucks = num_trucks,method = "plan")
    agent = DeliveryQAgent(env.observation_space,env.action_space,env.piece_space)

    #e,a,env_min = run_n_episodes(env,agent,"training_100_stops.gif")

    run_n_episodes(env,agent,f"training_{num_stops}_stops_{num_trucks}t_{iters}iter.gif",n_episodes = iters)

    # env.render()

    if os.path.exists("extraction.csv"):
        os.remove("extraction.csv")
    env.extract_csv("extraction.csv")

    
    
    #env_min.render()

    
    # ###Profiling for possible paralization###
    # profile.disable()
    # ps = pstats.Stats(profile)
    # ps.sort_stats('tottime', 'calls')
    # ps.print_stats(10)
    # #########################################
    
    


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)

