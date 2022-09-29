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
env = DeliveryEnvironment(n_stops = 1967, method = "plan")




env.render()

print(f"The first stop id: {env.stops}")
#print(f"Coordinates are: {env.x[env.stops[0]]} {env.y[env.stops[0]]}")


for i in range(4):
    env.step(i)
    print(f"Stops visited in step {i}: {env.stops}")

env.render()
