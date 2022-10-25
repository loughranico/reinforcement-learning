import csv
import pandas as pd
from collections import defaultdict

filename ="./data/pedido.csv"
 
# opening the file using "with"
# statement
'''with open(filename, 'r') as data:
    datadict = csv.DictReader(data)

    print(data["lonBase"])'''


'''deliveries = pd.read_csv(filename)


print(deliveries)'''


'''with open(filename) as f:
    r = csv.reader(f)
    d = defaultdict(list)
    for row in r:
        d[row[0]] = row[1:]
print(d["A127958B"])'''


'''
#env = DeliveryEnvironment(n_stops = 50)
#env = DeliveryEnvironment(n_stops = 2000, max_box = 1000)




env.render()

print(f"The first stop id: {env.stops}")

for i in range(4):
    env.step(i)
    print(f"Stops visited in step {i}: {env.stops}")

env.render()'''



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
# env = DeliveryEnvironment(n_stops = 1967, method = "plan")




# env.render()

# print(f"The first stop id: {env.stops}")
# #print(f"Coordinates are: {env.x[env.stops[0]]} {env.y[env.stops[0]]}")


# for i in range(4):
#     env.step(i)
#     print(f"Stops visited in step {i}: {env.stops}")

# env.render()

delivery_file ="./data/pedido.csv"
deliveries = pd.read_csv(delivery_file)

x_origin = deliveries["lonCarga"]
y_origin = deliveries["latCarga"]





## CDIST
xy_origin = np.column_stack([x_origin,y_origin])

            
# create projections, using a mean (lat, lon) for aeqd
lat_0, lon_0 = np.mean(np.append(xy_origin[:,0], xy_origin[:,0])), np.mean(np.append(xy_origin[:,1], xy_origin[:,1]))
proj = pyproj.Proj(proj='aeqd', lat_0=lat_0, lon_0=lon_0, x_0=lon_0, y_0=lat_0)
WGS84 = pyproj.Proj(init='epsg:4326')

# transform coordinates
projected_c1 = pyproj.transform(WGS84, proj, xy_origin[:,1], xy_origin[:,0])
projected_c1 = np.column_stack(projected_c1)

# calculate pairwise distances in km with both methods
sc_dist = cdist(projected_c1, projected_c1)

q_stops = sc_dist/1000 #Metres to KM

print(q_stops)



for x,y in deliveries["lonCarga","latCarga"]:
    print(x,y)
