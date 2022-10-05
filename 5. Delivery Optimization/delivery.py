# Base Data Science snippet
from datetime import datetime, timedelta
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from scipy.spatial.distance import cdist
import imageio
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import copy
import pandas as pd

import csv
from collections import defaultdict
import pyproj

from collections import defaultdict




plt.style.use("seaborn-dark")

import sys
sys.path.append("../")
from rl.agents.q_agent import QAgent




class DeliveryEnvironment(object):
    def __init__(self,n_stops = 10,max_box = 10,method = "distance",**kwargs):

        print(f"Initialized Delivery Environment with {n_stops} random stops")
        print(f"Target metric for optimization is {method}")

        # Initialization
        self.n_stops = n_stops
        self.action_space = self.n_stops
        self.observation_space = self.n_stops
        self.max_box = max_box
        self.stops = []
        self.method = method

        # List of deliveries. Unsure if also need the seen deliveries or just 
        self.to_go = []

        # Importing the data
        # Potentially need trucks in dict in the main when creating agents 
        truck_file ="./data/camion.csv"
        self.trucks = pd.read_csv(truck_file)
        '''with open(truck_file) as f:
            r = csv.reader(f)
            self.d = defaultdict(list)
            for row in r:
                self.d[row[0]] = row[1:]'''

        delivery_file ="./data/pedido.csv"
        self.deliveries = pd.read_csv(delivery_file)

        # Generate stops
        self._generate_constraints(**kwargs)
        self._generate_stops()
        if self.method == "plan":
            self._generate_trucks()
        self._generate_q_values()
        self.render()

        # Initialize first point
        self.reset()


    def _generate_constraints(self,box_size = 0.2,traffic_intensity = 5):

        '''if self.method == "traffic_box":

            x_left = np.random.rand() * (self.max_box) * (1-box_size)
            y_bottom = np.random.rand() * (self.max_box) * (1-box_size)

            x_right = x_left + np.random.rand() * box_size * self.max_box
            y_top = y_bottom + np.random.rand() * box_size * self.max_box

            self.box = (x_left,x_right,y_bottom,y_top)
            self.traffic_intensity = traffic_intensity '''
        #elif self.method == "plan":
            #TODO
        self.start_date = datetime.strptime("07:00:00 01/08/2022", "%H:%M:%S %d/%m/%Y")
        self.max_worktime = 540
        self.daily_worktime = 0
        self.timed_dels = []



    def _generate_stops(self):

        if self.method == "plan":
            self.x_origin = self.deliveries["lonCarga"]
            self.y_origin = self.deliveries["latCarga"]

            self.x_dest = self.deliveries["lonDescarga"]
            self.y_dest = self.deliveries["latDescarga"]


        else:
            if self.method == "traffic_box":

                points = []
                while len(points) < self.n_stops:
                    x,y = np.random.rand(2)*self.max_box
                    if not self._is_in_box(x,y,self.box):
                        points.append((x,y))

                xy = np.array(points)

            else:
                # Generate geographical coordinates
                xy = np.random.rand(self.n_stops,2)*self.max_box

            self.x = self.deliveries["lonDescarga"]
            self.y = self.deliveries["latDescarga"]
            '''self.x = xy[:,0]
            self.y = xy[:,1]'''

    def _generate_trucks(self):

        self.x_base = self.trucks["lonBase"]
        self.y_base = self.trucks["latBase"]
  
      
        # Defining the dict and passing 
        # lambda as default_factory argument
        self.truck_dict = defaultdict(dict)
        truck_file ="./data/camion.csv"
        with open(truck_file) as f:
            r = csv.reader(f)
            for nombre,lon,lat in r:
                if nombre != "idCamion":
                    self.truck_dict[nombre] = {"name":nombre,"lon":lon,"lat":lat,"daily_worktime":0}
        



    def _generate_q_values(self,box_size = 0.2):

        # Generate actual Q Values corresponding to time elapsed between two points
        if self.method in ["distance"]:
            xy = np.column_stack([self.x,self.y])
            self.q_stops = cdist(xy,xy)
        
        elif self.method == "plan":
            ## DO STUFF
            xy_dest = np.column_stack([self.x_dest,self.y_dest])
            xy_origin = np.column_stack([self.x_origin,self.y_origin])

            
            # create projections, using a mean (lat, lon) for aeqd
            lat_0, lon_0 = np.mean(np.append(xy_dest[:,0], xy_origin[:,0])), np.mean(np.append(xy_dest[:,1], xy_origin[:,1]))
            proj = pyproj.Proj(proj='aeqd', lat_0=lat_0, lon_0=lon_0, x_0=lon_0, y_0=lat_0)
            WGS84 = pyproj.Proj(init='epsg:4326')

            # transform coordinates
            projected_c1 = pyproj.transform(WGS84, proj, xy_dest[:,1], xy_dest[:,0])
            projected_c2 = pyproj.transform(WGS84, proj, xy_origin[:,1], xy_origin[:,0])
            projected_c1 = np.column_stack(projected_c1)
            projected_c2 = np.column_stack(projected_c2)

            # calculate pairwise distances in km with both methods
            sc_dist = cdist(projected_c1, projected_c2)

            self.q_stops = sc_dist/1000 #Metres to KM
            

        else:
            raise Exception("Method not recognized")
    
    def extract_csv(self,file_name):

        fieldnames = ['idCamion','idPedido', 'start_date', 'end_date']

        with open(file_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(fieldnames)

            # write multiple rows
            writer.writerows(self.timed_dels)
        
        
    

    def render(self,return_img = False):
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")

        # Show stops
        if self.method == "plan":
            ax.scatter(self.x_origin,self.y_origin,c = "red",s = 50)
            ax.scatter(self.x_dest,self.y_dest,c = "pink",s = 50)
            ax.scatter(self.x_base,self.y_base,c = "orange",s = 50)

            # Show START
            if len(self.stops)>0:
                xy = self._get_xy_del(initial = True)
                xytext = xy[0],xy[1]-0.005
                ax.annotate("START",xy=xy,xytext=xytext,weight = "bold")

            # Show itinerary
            if len(self.stops) > 1:
                for i in range(len(self.stops)):
                    x = [self.x_origin[i],self.x_dest[i]]
                    y = [self.y_origin[i],self.y_dest[i]]
                    
                    ax.plot(x,y,c = "blue",linewidth=1,linestyle="--")
                    '''if i == 0:
                        x_ini,y_ini = self._get_xy_del(initial = True)
                        xx = [x_ini,self.x_origin[i]]
                        yy = [y_ini,self.y_origin[i]]
                    else:
                        xx = [self.x_dest[i-1],self.x_origin[i]]
                        yy = [self.y_dest[i-1],self.y_origin[i]]
                    
                    
                    ax.plot(xx,yy,c = "green",linewidth=1,linestyle="-")'''

                
                # Annotate END
                xy = self._get_xy_del(initial = False)
                xytext = xy[0],xy[1]-0.005
                ax.annotate("END",xy=xy,xytext=xytext,weight = "bold")


        else:
            ax.scatter(self.x,self.y,c = "red",s = 50)


            # Show START
            if len(self.stops)>0:
                xy = self._get_xy(initial = True)
                xytext = xy[0]+0.1,xy[1]-0.05
                ax.annotate("START",xy=xy,xytext=xytext,weight = "bold")

            # Show itinerary
            if len(self.stops) > 1:
                ax.plot(self.x[self.stops],self.y[self.stops],c = "blue",linewidth=1,linestyle="--")
                
                # Annotate END
                xy = self._get_xy(initial = False)
                xytext = xy[0]+0.1,xy[1]-0.05
                ax.annotate("END",xy=xy,xytext=xytext,weight = "bold")


        plt.xticks([])
        plt.yticks([])
        
        if return_img:
            # From https://ndres.me/post/matplotlib-animated-gifs-easily/
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
        else:
            plt.show()



    def reset(self):

        # Stops placeholder
        self.stops = []
        self.timed_dels = []

        
        # Random first stop
        first_stop = np.random.randint(self.n_stops)
        # first_stop = 0
        self.stops.append(first_stop)

        return first_stop


    #def step(self,destination,truck):
    def step(self,destination):

        # Get current state
        state = self._get_state()
        new_state = destination

        # Get reward for such a move
        reward = self._get_reward(state,new_state)

        # Get reward for such a move
        delivery_kms = self._get_reward(new_state,new_state)

        # Append new_state to stops
        self.stops.append(destination)
        done = len(self.stops) == self.n_stops



        del_time = (reward+delivery_kms)/70*60
        unload_time = del_time + 90

        '''if self.truck_dict[truck].daily_worktime+del_time >= self.max_worktime:
            self.truck_dict[truck]. = del_time
            agent.start_date = agent.start_date.replace(hour=7, minute=0, second=0)
            agent.start_date += timedelta(days=1)

            end_date = agent.start_date + timedelta(minutes=unload_time)
        else:
            self.truck_dict[truck].daily_worktime += del_time
            end_date = agent.start_date + timedelta(minutes=unload_time)



        self.timed_dels.append([self.truck_dict[truck].nombre,self.deliveries.idPedido.iloc[new_state],agent.start_date,end_date])

        agent.start_date=end_date'''

        return new_state,reward,done
    

    def _get_state(self):
        return self.stops[-1]


    def _get_xy(self,initial = False):
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x,y

    def _get_xy_del(self,initial = False):
        if initial:
            x = self.x_base[0]
            y = self.y_base[0]
        else:
            x = self.x_dest[self._get_state()]
            y = self.y_dest[self._get_state()]

        return x,y


    def _get_reward(self,state,new_state):
        base_reward = self.q_stops[state,new_state]

        if self.method == "distance":
            return base_reward

        elif self.method == "plan":
            ## DO STUFF
            '''# Additional reward correspond to slowing down in traffic
            xs,ys = self.x[state],self.y[state]
            xe,ye = self.x[new_state],self.y[new_state]
            intersections = self._calculate_box_intersection(xs,xe,ys,ye,self.box)
            if len(intersections) > 0:
                i1,i2 = intersections
                distance_traffic = np.sqrt((i2[1]-i1[1])**2 + (i2[0]-i1[0])**2)
                additional_reward = distance_traffic * self.traffic_intensity * np.random.rand()
            else:
                additional_reward = np.random.rand()'''

            return base_reward

            







class DeliveryQAgent(QAgent):

    def __init__(self,*args,name = "",x_base = 0,y_base = 0,**kwargs):
        super().__init__(*args,**kwargs)
        self.reset_memory()
        self.name = name
        self.x_base = x_base
        self.y_base = y_base
        self.start_date = datetime.strptime("07:00:00 01/08/2022", "%H:%M:%S %d/%m/%Y")
        self.daily_worktime = 0
        

    def act(self,s):

        # Get Q Vector
        q = np.copy(self.Q[s,:])

        # Avoid already visited states
        q[self.states_memory] = -np.inf

        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            a = np.random.choice([x for x in range(self.actions_size) if x not in self.states_memory])

        return a


    def remember_state(self,s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []






def run_episode(env:DeliveryEnvironment,agent:DeliveryQAgent,verbose = 1):

    s = env.reset()
    agent.reset_memory()

    max_step = env.n_stops

    episode_reward = 0

    i = 0
    while i < max_step:

        # Remember the states
        agent.remember_state(s)

        # Choose an action
        a = agent.act(s)
        
        # Take the action, and get the reward from environment
        s_next,r,done = env.step(a)

        # Tweak the reward
        r = -1 * r
        
        if verbose: print(s_next,r,done)

        '''print(i)
        print("s = ",s)
        print("a = ",a)
        print("s_next = ",s_next)
        print()'''
        
        # Update our knowledge in the Q-table
        agent.train(s,a,r,s_next)
        
        # Update the caches
        episode_reward += r
        s = s_next
        
        # If the episode is terminated
        i += 1
        if done:
            break
            
    return env,agent,episode_reward




def run_n_episodes(env,agent,name="training.gif",n_episodes=1000,render_each=10,fps=10):

    # Store the rewards
    rewards = []
    imgs = []

    # env_min = copy.deepcopy(env)
        
    # Experience replay
    for i in tqdm(range(n_episodes)):

        # Run the episode
        env,agent,episode_reward = run_episode(env,agent,verbose = 0)
        '''if len(rewards)!=0:
            if max(rewards) < episode_reward:
                env_min = copy.deepcopy(env)'''
        rewards.append(episode_reward)
        
        if i % render_each == 0:
            img = env.render(return_img = True)
            imgs.append(img)

        

    # Show rewards
    plt.figure(figsize = (15,3))
    plt.title("Rewards over training")
    plt.plot(rewards)
    plt.show()

    # Save imgs as gif
    imageio.mimsave(name,imgs,fps = fps)

    return env,agent #,env_min



'''
Multi-agent approach doesn't work. At the very least, the approach tried was not useful. It lacked a centralised system that could communicate between agents.

Also multi-agent might not be appropriate because there is no real communication/interaction between agents to achieve the goal.
Main goals of MAS is cooperation or competition. Neither applies here

What is needed is one sole agent (planner) instead of multiple agents (trucks)




def run_episode_ma(env:DeliveryEnvironment,agents:List[DeliveryQAgent],verbose = 1):

    s = env.reset()

    for agent in agents:

        agent.reset_memory()

    max_step = env.n_stops
    
    episode_reward = 0
    
    i = 0
    agent_count = 0
    while i < max_step:

        # Remember the states
        agents[agent_count].remember_state(s)

        # Choose an action
        a = agents[agent_count].act(s)
        
        # Take the action, and get the reward from environment
        s_next,r,done = env.step(a,agents[agent_count])

        # Tweak the reward
        r = -1 * r
        
        if verbose: print(s_next,r,done)
        
        # Update our knowledge in the Q-table
        agents[agent_count].train(s,a,r,s_next)
        
        # Update the caches
        episode_reward += r
        s = s_next
        
        # If the episode is terminated
        i += 1
        agent_count += 1
        agent_count %= len(agents)
        if done:
            break
            
    return env,agents,episode_reward




def run_n_episodes_ma(env,agents,name="training.gif",n_episodes=1000,render_each=10,fps=10):

    # Store the rewards
    rewards = []
    imgs = []

    # env_min = copy.deepcopy(env)
        
    # Experience replay
    for i in tqdm(range(n_episodes)):

        # Run the episode
        env,agents,episode_reward = run_episode_ma(env,agents,verbose = 0)
        if len(rewards)!=0:
            if max(rewards) < episode_reward:
                env_min = copy.deepcopy(env)
        rewards.append(episode_reward)
        
        if i % render_each == 0:
            img = env.render(return_img = True)
            imgs.append(img)

        

    # Show rewards
    plt.figure(figsize = (15,3))
    plt.title("Rewards over training")
    plt.plot(rewards)
    plt.show()

    # Save imgs as gif
    imageio.mimsave(name,imgs,fps = fps)

    return env,agents #,env_min'''