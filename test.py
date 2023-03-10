import pickle
import torch as T
import numpy as np
import collections

import common_test, ppo
from env import MEDAEnv
from map import MakeMap
from map import Symbols

import warnings
warnings.filterwarnings("ignore")

def _is_touching(dstate, obj, map, dsize):
		i = 0
		while True:
			j = 0
			while True:
				if map[dstate[1]+j][dstate[0]+i] == obj:
					return True
				j += 1
				if j == dsize:
					break
			i += 1
			if i == dsize:
				break

		return False

def _compute_shortest_route(w, h, dsize, symbols,map, start):
	queue = collections.deque([[start]])
	seen = set([start])
#		print(self.map)
	while queue:
		path = queue.popleft()
#			print(path)
		x, y = path[-1]
		if _is_touching((x,y), symbols.Goal, map, dsize):
			return path
		for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
			if 0 <= x2 < (w-dsize+1) and 0 <= y2 < (h-dsize+1) and \
			(_is_touching((x2,y2), symbols.Dynamic_module, map, dsize) == False) and\
			(_is_touching((x2,y2), symbols.Static_module, map, dsize) == False) and\
			(x2, y2) not in seen:
				queue.append(path + [(x2, y2)])
				seen.add((x2, y2))
#		print("Bad map")
#		print(self.map)
	return False

if __name__ == "__main__":
	T.manual_seed(42)
	###### Set params ##########

	W = 5
	H = 8
	DSIZE = 1
	S_MODULES = 2
	D_MODULES = 2
	N_EPOCH = 63

	############################
	ENV_NAME = str(W)+str(H)+str(DSIZE)+str(S_MODULES)+str(D_MODULES)+ "/" + str(N_EPOCH)

	test_result = common_test.test(ENV_NAME, W, H, DSIZE, S_MODULES, D_MODULES)

	print(test_result)