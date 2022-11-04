# Base Data Science snippet
from collections import defaultdict
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




plt.style.use("seaborn-dark")

import sys
sys.path.append("../")
from rl.agents.q_agent import QAgent




class DeliveryEnvironment(object):
    def __init__(self,n_stops = 10,n_trucks = 2,max_box = 10,method = "distance",data_size = "tiny",**kwargs):

        print(f"Initialized Delivery Environment with {n_stops} deliveries")
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

        if data_size == "tiny":
            self.data_folder = "./data_tiny/"
        elif data_size == "small":
            self.data_folder = "./data_small/"
        elif data_size == "medium":
            self.data_folder = "./data_med/"
        elif data_size == "large":
            self.data_folder = "./data_large/"
        elif data_size == "tiny2":
            self.data_folder = "./data_tiny2/"
        elif data_size == "small2":
            self.data_folder = "./data_small2/"
        elif data_size == "medium2":
            self.data_folder = "./data_med2/"
        elif data_size == "large2":
            self.data_folder = "./data_large2/"
        elif data_size == "total":
            self.data_folder = "./data/"
        else:
            raise Exception("Data size not recognized")


        # Importing the data
        # Potentially need trucks in dict in the main when creating agents 
        truck_file =self.data_folder+"camion.csv"
        print(truck_file)
        self.trucks = pd.read_csv(truck_file)

        delivery_file =self.data_folder+"pedido.csv"
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
        self.timed_dels = []



    def _generate_stops(self):

        if self.method == "plan":
            self.x_origin = self.deliveries["lonCarga"]
            self.y_origin = self.deliveries["latCarga"]

            self.x_dest = self.deliveries["lonDescarga"]
            self.y_dest = self.deliveries["latDescarga"]

            # Defining the dict and passing 
            # lambda as default_factory argument
            self.pedido_dict = defaultdict(dict)
            self.pedido_names = []
            deliveries_file =self.data_folder+"pedido.csv"
            with open(deliveries_file) as f:
                r = csv.reader(f)
                i=0
                for idPedido,lon,lat,lonDes,latDes,categoria,prio,start_time,end_time,last in r:
                    if idPedido != "idPedido":
                        self.pedido_names.append(idPedido)
                        self.pedido_dict[idPedido] = {"num":i,
                                                    "name":idPedido,
                                                    "lonCarga":lon,
                                                    "latCarga":lat,
                                                    "lonDescarga":lonDes,
                                                    "latDescarga":latDes,
                                                    "categoria":categoria,
                                                    "start_time":start_time,
                                                    "end_time":end_time,
                                                    "lastDay":last,
                                                    "daily_worktime":0,
                                                    "start_date":datetime.strptime("07:00:00 01/08/2022", "%H:%M:%S %d/%m/%Y")}
                        i+=1

            


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

  
        self.x_dest = np.concatenate((self.x_dest, self.x_base), axis=None)
        self.y_dest = np.concatenate((self.y_dest, self.y_base), axis=None)
        self.x_origin = np.concatenate((self.x_origin, self.x_base), axis=None)
        self.y_origin = np.concatenate((self.y_origin, self.y_base), axis=None)
      
        # Defining the dict and passing 
        # lambda as default_factory argument
        self.truck_dict = defaultdict(dict)
        self.truck_names = []
        truck_file =self.data_folder+"camion.csv"
        with open(truck_file) as f:
            r = csv.reader(f)
            i=0
            for nombre,lon,lat in r:
                if nombre != "idCamion":
                    self.truck_names.append(nombre)
                    self.truck_dict[nombre] = {"num":i,"name":nombre,"lon":lon,"lat":lat,"daily_worktime":0,"start_date":datetime.strptime("07:00:00 01/08/2022", "%H:%M:%S %d/%m/%Y")}
                    i+=1

        self.camion_pedidos = defaultdict(list)
        order_file =self.data_folder+"camiones_pedido.csv"
        with open(order_file) as f:
            r = csv.reader(f)

            for idPedido,idCamion in r:
                if idPedido != "idPedido":
                    self.camion_pedidos[idCamion].append(self.pedido_dict[idPedido]["num"])
        
        self.pedido_camiones = defaultdict(list)
        with open(order_file) as f:
            r = csv.reader(f)

            for idPedido,idCamion in r:
                if idPedido != "idPedido":
                    self.pedido_camiones[idPedido].append(self.truck_dict[idCamion]["num"])
        
        self.paradas = defaultdict(list)
        rest_file =self.data_folder+"paradas.csv"
        with open(rest_file) as f:
            r = csv.reader(f)

            for idCamion,descanso in r:
                if idCamion != "idCamion":
                    self.paradas[idCamion].append(datetime.strptime("00:00:00 "+descanso, "%H:%M:%S %m/%d/%Y").date())
        
        self.last_stop = defaultdict(int)
        



    def _generate_q_values(self):

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
            WGS84 = pyproj.Proj('epsg:4326')

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

        fieldnames = ['idCamion','idPedido', 'start_date', 'end_date','late']

        with open(file_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(fieldnames)

            # write multiple rows
            writer.writerows(self.timed_dels)
        
        
    

    def render(self,return_img = False):
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops. Cost: "+str(self.reward))

        # Show stops
        if self.method == "plan":
            ax.scatter(self.x_origin,self.y_origin,c = "red",s = 50)
            ax.scatter(self.x_dest,self.y_dest,c = "pink",s = 50)
            ax.scatter(self.x_base,self.y_base,c = "orange",s = 50)

            # Show START
            if len(self.stops)>0:
                for i in range(self.n_trucks):

                    # x = self.x[i+self.n_stops]
                    # y = self.y[i+self.n_stops]
                    x = self.x_base[i]
                    y = self.y_base[i]
                    xytext = x+0.01,y-0.005
                    ax.annotate("START",xy=(x,y),xytext=xytext,weight = "bold")

            # Show itinerary
            if len(self.stops) > self.n_trucks:
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

                    '''for i in range(self.n_trucks):
                t_stops = [z[0] for z in self.stops if z[1]==i ]
                ax.plot(self.x[t_stops],self.y[t_stops],linewidth=1,linestyle="-")'''

                
                # # Annotate END
                # xy = self._get_xy_del(initial = False)
                # xytext = xy[0],xy[1]-0.005
                # ax.annotate("END",xy=xy,xytext=xytext,weight = "bold")


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

        for key in self.truck_dict.keys():
            self.truck_dict[key]["daily_worktime"] = 0
            self.truck_dict[key]["start_date"] = datetime.strptime("07:00:00 01/08/2022", "%H:%M:%S %d/%m/%Y")

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


    #def step(self,destination,truck):
    def step(self,destination):

        # Get current state
        # state = self._get_state()
        new_state = destination

        prev_stop = self.last_stop[new_state[1]]

        # Get reward for such a move
        reward = self._get_reward(prev_stop,new_state[0])
        self.reward += reward

        # Get reward for such a move
        delivery_kms = self._get_reward(new_state[0],new_state[0])

        # Append new_state to stops
        self.stops.append(destination)
        self.last_stop[new_state[1]] = new_state[0]
        done = len(self.stops) == (self.n_stops+self.n_trucks)



        # Keeping times etc.
        # Tiempos de tránsito 
        rew_speed = 70
        deliv_speed = 70
        if reward > 60: 
            rew_speed = 100
        if delivery_kms > 60: 
            deliv_speed = 100

        rew_time = reward/rew_speed*60
        deliv_time = delivery_kms/deliv_speed*60
        
        del_time = rew_time + deliv_time


        # Tiempos de carga
        idPedido = self.deliveries.idPedido.iloc[new_state[0]]
        cat = self.pedido_dict[idPedido]["categoria"]

        t_carga = 45
        if cat == "M1":
            t_carga = 35
        elif cat == "M2":
            t_carga = 20
        elif cat == "M3":
            t_carga = 15
        
        t_descarga = t_carga

        unload_time = del_time + t_descarga + t_carga


        # Truck ID
        truck = self.truck_names[new_state[1]]

        if (self.truck_dict[truck]["daily_worktime"]+del_time) >= self.max_worktime:
            self.truck_dict[truck]["daily_worktime"] = del_time
            self.truck_dict[truck]["start_date"] = self.truck_dict[truck]["start_date"].replace(hour=7, minute=0, second=0)

            # Paradas
            if self.truck_dict[truck]["start_date"].date() + timedelta(days=1) in self.paradas[truck]:
                if self.truck_dict[truck]["start_date"].date() + timedelta(days=2) in self.paradas[truck]:
                    self.truck_dict[truck]["start_date"] += timedelta(days=3)
                else:
                    self.truck_dict[truck]["start_date"] += timedelta(days=2)
            else:
                self.truck_dict[truck]["start_date"] += timedelta(days=1)

            end_date = self.truck_dict[truck]["start_date"] + timedelta(minutes=unload_time)
        else:
            if self.truck_dict[truck]["start_date"].date() in self.paradas[truck]:
                if self.truck_dict[truck]["start_date"].date() + timedelta(days=1) in self.paradas[truck]:
                    self.truck_dict[truck]["start_date"] += timedelta(days=2)
                else:
                    self.truck_dict[truck]["start_date"] += timedelta(days=1)
            self.truck_dict[truck]["daily_worktime"] += del_time
            end_date = self.truck_dict[truck]["start_date"] + timedelta(minutes=unload_time)


        # Penalising lateness
        late = 0
        idPedido = self.pedido_names[new_state[0]]
        pedido = self.pedido_dict[idPedido]
        

        # Ensure delivery in window
        start_window = datetime.strptime(pedido["start_time"]+":00 "+str(end_date.year)+"-"+str(end_date.month)+"-"+str(end_date.day), "%H:%M:%S %Y-%m-%d")
        end_window = datetime.strptime(pedido["end_time"]+":00 "+str(end_date.year)+"-"+str(end_date.month)+"-"+str(end_date.day), "%H:%M:%S %Y-%m-%d")
        waiting_time = 0

        if start_window.time()>end_date.time():
            waiting_time = (start_window-end_date).seconds / 60
            reward += 10
            
        elif end_window.time()<end_date.time():
            late_time = (end_date-end_window).seconds / 60
            start_window += timedelta(days=1)
            if start_window.date() + timedelta(days=1) in self.paradas[truck]:
                if start_window.date() + timedelta(days=2) in self.paradas[truck]:
                    start_window += timedelta(days=3)
                else:
                    start_window += timedelta(days=2)
            else:
                start_window += timedelta(days=1)
            waiting_time = (end_date-start_window).seconds / 60
            reward += 10
            end_date = start_window
        
        # Penalising lateness
        limit_date = datetime.strptime("23:59:59 "+pedido["lastDay"], "%H:%M:%S %Y-%m-%d")
        if limit_date < end_date:
            delta =  end_date - limit_date
            reward += 100*(delta.days+1)
            late = delta.days+1




            



        self.timed_dels.append([truck,self.deliveries.idPedido.iloc[new_state[0]],self.truck_dict[truck]["start_date"],end_date,late])

        self.truck_dict[truck]["start_date"]=end_date

        return new_state,reward,done
    

    def _get_state(self):
        return self.stops[-1]


    def _get_xy(self,initial = False):
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
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

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.reset_memory()
        # self.start_date = datetime.strptime("07:00:00 01/08/2022", "%H:%M:%S %d/%m/%Y")
        # self.daily_worktime = 0

    def act(self,env:DeliveryEnvironment,s):

        if s[0] >= self.actions_size:
            truck_number = s[0] - self.actions_size
            # Get Q Vector
            q = np.copy(self.Q[self.actions_size,:,truck_number])
            

            # Avoid already visited states
            q[self.states_memory] = -np.inf

            if np.random.rand() > self.epsilon:
                a = (np.argmax(q),truck_number)
            else:
                # act_size = np.array(range(self.actions_size))
                idCamion = env.truck_names[s[1]]
                act_size = np.array(env.camion_pedidos[idCamion])
                stat_mem = np.array(self.states_memory)

                a_a = np.setdiff1d(act_size, stat_mem)

                #a_a = [x for x in range(self.actions_size) if x not in self.states_memory]
                i = np.random.randint(0,len(a_a))
                a = (a_a[i],truck_number)

        else:
            # Get Q Vector
            q = np.copy(self.Q[s[0],:,:])
            

            # Avoid already visited states
            q[self.states_memory] = -np.inf



            if np.random.rand() > self.epsilon:
                #a = np.argmax(q)
                # a = np.unravel_index(np.argmax(q, axis=None), q.shape)

                winner = np.argwhere(q == np.amax(q))

                if len(winner) > 1:
                    # i = random.randrange(len(winner))
                    i = np.random.randint(len(winner))
                    a = (winner[i][0],winner[i][1])
                else:
                    a = np.unravel_index(np.argmax(q, axis=None), q.shape)
                
            else:
                
                
                act_size_init = np.array(range(self.actions_size))
                stat_mem = np.array(self.states_memory)
                available = np.setdiff1d(act_size_init, stat_mem)


                i = np.random.randint(0,len(available))

                idPedido = env.pedido_names[i]
                ava_trucks = np.array(env.pedido_camiones[idPedido])
                

                


                j = np.random.randint(0,len(ava_trucks))
                a = (i,ava_trucks[j])
           
        return a


    def remember_state(self,s):
        if s[0] >= self.actions_size:
            self.states_memory.append(self.actions_size)
        else:
            self.states_memory.append(s[0])

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
        a = agent.act(env,s)
        
        # Take the action, and get the reward from environment
        s_next,r,done = env.step(a)

        # Tweak the reward
        r = -1 * r
        
        if verbose: print(s_next,r,done)
        
        # Update our knowledge in the Q-table
        # Check if leaving the base or not
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




def run_n_episodes(env,agent,name="training.gif",n_episodes=1000,render_each=10,fps=10):

    # Store the rewards
    rewards = []
    imgs = []

    env_min = copy.deepcopy(env)
    prev_best = -np.inf
        
    # Experience replay
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



