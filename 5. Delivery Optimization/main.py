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
    

    num_stops = int(args[0])
    num_trucks = int(args[1])
    iters = int(args[2])
    # num_replan = int(args[3])
    
    env = DeliveryEnvironment(n_stops = num_stops,n_trucks = num_trucks,method = "distance")
    agent = DeliveryQAgent(env.observation_space,env.action_space,env.piece_space)

    e,a,env_min = run_n_episodes(env,agent,f"training_{num_stops}_stops_{num_trucks}t_{iters}iter.gif",n_episodes = iters)

    # run_n_episodes(env,agent,f"training_{num_stops}_stops_{num_trucks}t_{iters}iter.gif",n_episodes = iters)

    # print(agent.Q)

    env_min.render()
    env.render()

    # print()
    # print("Replan")

    # agent.expand(num_replan)
    # env.generate_replan(num_replan)
    # env._generate_q_values()

    # env.render()
    # print(agent.Q)
    # run_n_episodes(env,agent,f"training_{num_stops+num_replan}_stops_{num_trucks}t_{iters}iter.gif",n_episodes = iters)

    # env.render()

    
    # ###Profiling for possible paralization###
    # profile.disable()
    # ps = pstats.Stats(profile)
    # ps.sort_stats('tottime', 'calls')
    # ps.print_stats(10)
    # #########################################
    
    


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)