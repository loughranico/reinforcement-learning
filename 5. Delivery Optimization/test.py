import numpy as np
from scipy.spatial.distance import cdist

n_stops = 4
n_trucks = 2
max_box = 10



xy = np.random.rand(n_stops,2)*max_box

x = xy[:,0]
y = xy[:,1]

xy_base = np.random.rand(n_trucks,2)*max_box

x_base = xy_base[:,0]
y_base = xy_base[:,1]



x_conc = np.concatenate((x, x_base), axis=None)
y_conc = np.concatenate((y, y_base), axis=None)
xy = np.column_stack([x_conc,y_conc])
q_stops = cdist(xy,xy)
print("Initial")
print(q_stops)

q_stops_2 = np.copy(q_stops)
q_stops_3 = np.copy(q_stops)
q_stops_4 = np.copy(q_stops)
print()


q_stops_2[n_stops:,n_stops:] = -np.inf
print("Bottom Corner")
print(q_stops_2)
print()

q_stops_3[n_stops:] = -np.inf
print("[n_stops:]")
print(q_stops_3)
print()

q_stops_3[1,0] = -np.inf
print("[1,0] = inf")
print(q_stops_3)
print()


q_stops_4[:,n_stops:] = -np.inf
print("[:,n_stops:]")
print(q_stops_4)
print()