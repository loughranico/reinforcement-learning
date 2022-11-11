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

def title_csv():

        fieldnames = ['dataset','km', 'tiempo', 'pedidosTardes','experimento']
        file_name = 'RL_benchmark_reduced.csv'

        with open(file_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(fieldnames)

def results_csv(dataset,km, tiempo, pedidosTardes,experimento):

        file_name = 'RL_benchmark_reduced.csv'

        with open(file_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)


            # write multiple rows
            writer.writerows(self.timed_dels)

def main(args):
    
    # ### Profiling ###
    # profile = cProfile.Profile()
    # profile.enable()
    # #################
    

    num_stops = 1967
    num_trucks = 107
    iters = int(args[0])
    data_size = str(args[1])
    experiment = str(args[2])

    if data_size == "tiny":
        num_stops = 50
        num_trucks = 5
        data_folder = "./data_tiny/"
    elif data_size == "small":
        num_stops = 100
        num_trucks = 10
        data_folder = "./data_small/"
    elif data_size == "medium":
        num_stops = 500
        num_trucks = 20
        data_folder = "./data_med/"
    elif data_size == "large":
        num_stops = 750
        num_trucks = 35
        data_folder = "./data_large/"
    elif data_size == "tiny2":
        num_stops = 50
        num_trucks = 10
        data_folder = "./data_tiny2/"
    elif data_size == "small2":
        num_stops = 100
        num_trucks = 20
        data_folder = "./data_small2/"
    elif data_size == "medium2":
        num_stops = 500
        num_trucks = 35
        data_folder = "./data_med2/"
    elif data_size == "large2":
        num_stops = 750
        num_trucks = 50
        data_folder = "./data_large2/"
    elif data_size == "total":
        num_stops = 1967
        num_trucks = 107
        data_folder = "./data/"
    else:
        raise Exception("Data size not recognized")
    
    env = DeliveryEnvironment(n_stops = num_stops,n_trucks = num_trucks,method = "plan",data_size = data_size)
    agent = DeliveryQAgent(env.observation_space,env.action_space,env.piece_space)

    e,a,env_min,exec_time = run_n_episodes(env,agent,f"training_{num_stops}_stops_{num_trucks}t_{iters}iter.gif",n_episodes = iters)

    # run_n_episodes(env,agent,f"training_{num_stops}_stops_{num_trucks}t_{iters}iter.gif",n_episodes = iters)

    # env.render()

    if os.path.exists(data_folder+"extraction_"+data_size+"_"+experiment+".csv"):
        os.remove(data_folder+"extraction_"+data_size+"_"+experiment+".csv")
    env_min.extract_csv(data_folder+"extraction_"+data_size+"_"+experiment+".csv")

    if not os.path.exists('RL_benchmark_reduced.csv'):
        title_csv()

    results_csv(data_size,env_min.reward,exec_time,env_min.late_deliveries,experiment)

    
    
    # env_min.render()

    
    # ###Profiling for possible paralization###
    # profile.disable()
    # ps = pstats.Stats(profile)
    # ps.sort_stats('tottime', 'calls')
    # ps.print_stats(10)
    # #########################################
    



if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)

