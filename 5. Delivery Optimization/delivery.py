# Base Data Science snippet
from collections import defaultdict
import random
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

plt.style.use("seaborn-dark")

import sys
sys.path.append("../")
from rl.agents.q_agent import QAgent




class DeliveryEnvironment(object):
    def __init__(self,n_stops = 10,n_trucks = 2,max_box = 10,method = "distance",**kwargs):

        print(f"Initialized Delivery Environment with {n_stops} random stops and {n_trucks} trucks")
        print(f"Target metric for optimization is {method}")

        # Initialization
        self.n_stops = n_stops
        self.n_trucks = n_trucks
        self.action_space = self.n_stops
        self.observation_space = self.n_stops
        self.piece_space = self.n_trucks
        self.max_box = max_box
        self.stops = []
        self.method = method
        self.reward = 0
        np.random.seed(10)

        

        # Generate stops
        self._generate_stops()
        self._generate_trucks()
        self._generate_q_values()
        self.render()

        # Initialize first point
        self.reset()

        


    def _generate_stops(self):

        xy = np.random.rand(self.n_stops,2)*self.max_box

        self.x = xy[:,0]
        self.y = xy[:,1]
    
    def generate_replan(self,num_replan):
        # Generate geographical coordinates
        self.num_replan = num_replan
        xy = np.random.rand(self.num_replan,2)*self.max_box

        self.x = np.concatenate((self.x, xy[:,0]), axis=None)
        self.y = np.concatenate((self.y, xy[:,1]), axis=None)

    def _generate_trucks(self):

        xy = np.random.rand(self.n_trucks,2)*self.max_box

        self.x_base = xy[:,0]
        self.y_base = xy[:,1]

        self.x = np.concatenate((self.x, self.x_base), axis=None)
        self.y = np.concatenate((self.y, self.y_base), axis=None)

        self.last_stop = defaultdict(int)




    def _generate_q_values(self,box_size = 0.2):

        # Generate actual Q Values corresponding to time elapsed between two points
        if self.method in ["distance","traffic_box"]:
            
            xy = np.column_stack([self.x,self.y])
            self.q_stops = cdist(xy,xy)
            # self.q_stops[:,self.n_stops:] = -np.inf
        else:
            raise Exception("Method not recognized")
    

    def render(self,return_img = False):
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops. Cost: "+str(self.reward))

        # Show stops
        ax.scatter(self.x[:self.n_stops],self.y[:self.n_stops],c = "black",s = 50)
        ax.scatter(self.x[self.n_stops:],self.y[self.n_stops:],c = "orange",s = 50)

        # Show START
        if len(self.stops)>0:
            for i in range(self.n_trucks):

                # xy = self._get_xy(initial = True)
                x = self.x[i+self.n_stops]
                y = self.y[i+self.n_stops]
                xytext = x+0.1,y-0.05
                ax.annotate("START",xy=(x,y),xytext=xytext,weight = "bold")

        # Show itinerary
        if len(self.stops) > self.n_trucks:
            '''zero_stops = [z[0] for z in self.stops if z[1]==0 ]
            one_stops = [z[0] for z in self.stops if z[1]==1 ]
            two_stops = [z[0] for z in self.stops if z[1]==2 ]
            ax.plot(self.x[zero_stops],self.y[zero_stops],c = "blue",linewidth=1,linestyle="--")

            
            ax.plot(self.x[one_stops],self.y[one_stops],c = "red",linewidth=1,linestyle="--")

            
            ax.plot(self.x[two_stops],self.y[two_stops],c = "black",linewidth=1,linestyle="--")'''

            for i in range(self.n_trucks):
                t_stops = [z[0] for z in self.stops if z[1]==i ]
                ax.plot(self.x[t_stops],self.y[t_stops],linewidth=1,linestyle="-")
            
            # Annotate END
            # xy = self._get_xy(initial = False)
            # xytext = xy[0]+0.1,xy[1]-0.05
            # ax.annotate("END",xy=xy,xytext=xytext,weight = "bold")


        if hasattr(self,"box"):
            left,bottom = self.box[0],self.box[2]
            width = self.box[1] - self.box[0]
            height = self.box[3] - self.box[2]
            rect = Rectangle((left,bottom), width, height)
            collection = PatchCollection([rect],facecolor = "red",alpha = 0.2)
            ax.add_collection(collection)


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

        # Random first stop
        for i in range(self.n_trucks):
            f_stop = i + self.n_stops
            f_truck = i
            self.stops.append((f_stop,f_truck))
            self.last_stop[f_truck] = f_stop
        random_truck = np.random.randint(self.n_trucks)
        truck_stop = random_truck + self.n_stops
        self.reward = 0

        return (truck_stop,random_truck)


    def step(self,destination):

        # Get current state
        # state = self._get_state()
        new_state = destination

        prev_stop = self.last_stop[new_state[1]]

        # Get reward for such a move
        reward = self._get_reward(prev_stop,new_state[0])
        self.reward += reward

        # Append new_state to stops
        self.stops.append(destination)
        self.last_stop[new_state[1]] = new_state[0]
        done = len(self.stops) == (self.n_stops+self.n_trucks)

        return new_state,reward,done
    

    def _get_state(self):
        return self.stops[-1]


    def _get_xy(self,initial = False):
        state = self.stops[0][0] if initial else self._get_state()[0]
        x = self.x[state]
        y = self.y[state]
        return x,y


    def _get_reward(self,state,new_state):
        base_reward = self.q_stops[state,new_state]

        if self.method == "distance":
            return base_reward
        elif self.method == "time":
            return base_reward + np.random.randn()
        elif self.method == "traffic_box":

            # Additional reward correspond to slowing down in traffic
            xs,ys = self.x[state],self.y[state]
            xe,ye = self.x[new_state],self.y[new_state]
            intersections = self._calculate_box_intersection(xs,xe,ys,ye,self.box)
            if len(intersections) > 0:
                i1,i2 = intersections
                distance_traffic = np.sqrt((i2[1]-i1[1])**2 + (i2[0]-i1[0])**2)
                additional_reward = distance_traffic * self.traffic_intensity * np.random.rand()
            else:
                additional_reward = np.random.rand()

            return base_reward + additional_reward







def run_episode(env,agent,verbose = 1):

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

        
        
        # Update our knowledge in the Q-table
        if s[0] >= env.n_stops:
            agent.train(env.n_stops,a,r,s_next[0])
        else:
            agent.train(s[0],a,r,s_next[0])
        
        # Update the caches
        episode_reward += r
        s = s_next
        
        # If the episode is terminated
        i += 1
        if done:
            break
            
    return env,agent,episode_reward






class DeliveryQAgent(QAgent):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.reset_memory()

    def act(self,s):

        if s[0] >= self.actions_size:
            truck_number = s[0] - self.actions_size
            # Get Q Vector
            q = np.copy(self.Q[self.actions_size,:,truck_number])
            

            # Avoid already visited states
            q[self.states_memory] = -np.inf

            if np.random.rand() > self.epsilon:
                a = (np.argmax(q),truck_number)
            else:
                act_size = np.array(range(self.actions_size))
                stat_mem = np.array(self.states_memory)

                a_a = np.setdiff1d(act_size, stat_mem)

                #a_a = [x for x in range(self.actions_size) if x not in self.states_memory]
                i = np.random.randint(0,len(a_a))
                a = (a_a[i],truck_number)

        else:



            # Get Q Vector
            q = np.copy(self.Q[s[0],:,:])
            

            # Avoid already visited states
            # print(self.states_memory)
            # print(q[self.states_memory])
            q[self.states_memory] = -np.inf



            if np.random.rand() > self.epsilon:
                #a = np.argmax(q)
                # a = np.unravel_index(np.argmax(q, axis=None), q.shape)
                # print(np.argmax(q, axis=None), q.shape)
                # winner = np.argwhere(q == np.amax(q))
                # print(winner)
                # print(winner[0])
                # print(winner[0][0])
                # print(winner.flatten().tolist())
                # print(len(winner))
                # print(a)
                # print()

                winner = np.argwhere(q == np.amax(q))

                if len(winner) > 1:
                    # i = random.randrange(len(winner))
                    i = np.random.randint(len(winner))
                    a = (winner[i][0],winner[i][1])
                else:
                    a = np.unravel_index(np.argmax(q, axis=None), q.shape)

                
                
                '''if a == (0,0):
                    print("q[0,0] = ",q[0,0])
                    
                    print("q[0,1] = ",q[0,1])'''
            else:
                '''available_actions = [x for x in self.actions if x[0] not in self.states_memory]
                i = np.random.choice(len(available_actions))
                a = available_actions[i]
                
                # 30 day execution
                '''



                #############################


                '''a_a = [x for x in range(self.actions_size) if x not in self.states_memory]
                i = np.random.randint(0,len(a_a))
                j = np.random.randint(0,self.piece_size)
                a = (a_a[i],j)
                
                # 15 hour execution
                '''

                #############################

            
                act_size = np.array(range(self.actions_size))
                stat_mem = np.array(self.states_memory)

                a_a = np.setdiff1d(act_size, stat_mem)

                #a_a = [x for x in range(self.actions_size) if x not in self.states_memory]
                i = np.random.randint(0,len(a_a))
                j = np.random.randint(0,self.piece_size)
                a = (a_a[i],j)
                


        return a


    def remember_state(self,s):
        if s[0] >= self.actions_size:
            self.states_memory.append(self.actions_size)
        else:
            self.states_memory.append(s[0])

    def reset_memory(self):
        self.states_memory = []



def run_n_episodes(env,agent,name="training.gif",n_episodes=1000,render_each=10,fps=10):

    # Store the rewards
    rewards = []
    imgs = []

    # Experience replay
    env_min = copy.deepcopy(env)
    prev_best = -np.inf
        
    for i in tqdm(range(n_episodes)):

        # Run the episode
        env,agent,episode_reward = run_episode(env,agent,verbose = 0)
        if len(rewards)!=0:
            if prev_best < episode_reward:
                env_min = copy.deepcopy(env)
                prev_best = episode_reward
        rewards.append(episode_reward)
        
        '''if i % render_each == 0:
            img = env.render(return_img = True)
            imgs.append(img)'''

        

    # Show rewards
    plt.figure(figsize = (15,3))
    plt.title("Rewards over training")
    plt.plot(rewards)
    plt.show()

    # # Save imgs as gif
    # imageio.mimsave(name,imgs,fps = fps)

    return env,agent,env_min