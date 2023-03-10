import os
import csv
import pickle
import collections
import torch as T
import numpy as np

import common, ppo
from env import MEDAEnv
from map import MakeMap
from map import Symbols

###### Set params ##########
TOTAL_GAMES = 100
W = 24
H = 0
DSIZE = 1
S_MODULES = 0
D_MODULES = 57

############################

dir_name = "testmaps/%sx%s/%s/%s,%s"%(W , H, DSIZE, S_MODULES, D_MODULES)
bfile_name = "%s/map.pkl"%(dir_name)
cfile_name = "%s/map.csv"%(dir_name)

if not os.path.exists(dir_name):
	os.makedirs(dir_name)

data = {}

map_symbols = Symbols()
mapclass = MakeMap(w=W,h=H,dsize=DSIZE,s_modules=S_MODULES,d_modules=D_MODULES)

for i in range(TOTAL_GAMES):
	map = mapclass.gen_random_map()
	data[i] = map

save_file = open(bfile_name, "wb")
pickle.dump(data, save_file)

with open(cfile_name, 'w') as csv_file:  
	writer = csv.writer(csv_file)
	for key, value in data.items():
		writer.writerow([key, value])

csv_file.close()
save_file.close()
