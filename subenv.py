import gym
import math
import random
import numpy as np
from enum import IntEnum

from map import MakeMap
from map import Symbols

class Actions(IntEnum):
	U = 0
	R = 1
	D = 2
	L = 3

class MEDAEnv(gym.Env):
	def __init__(self, w=8, h=8, dsize=2, s_modules=2, d_modules=2, test_flag=False):
		super(MEDAEnv, self).__init__()
		assert w>0 and h>0 and dsize>0
		assert 0<=s_modules and 0<=d_modules
		self.w = w
		self.h = h
		self.dsize = dsize
		self.actions = Actions
		self.action_space = len(self.actions)
		self.observation_space = (3, w, h)
		self.n_steps = 0
		self.max_step = 5*(w+h)

		self.state = (0,0)
		self.goal = (w-1, h-1)

		self.map_symbols = Symbols()
		self.mapclass = MakeMap(w=self.w,h=self.h,dsize=self.dsize,s_modules=s_modules,d_modules=d_modules)
		self.map = self.mapclass.gen_random_map()

		self.test_flag = test_flag

		self.dynamic_flag = 0
		self.dynamic_state = (0,0)

	def reset(self, test_map=None):
		self.n_steps = 0
		self.state = (0, 0)

		if self.test_flag == False:
			self.map = self.mapclass.gen_random_map()
		else:
			self.map = test_map

		obs = self._get_obs()

		return obs

	def step(self, action):
		done = False
		message = None
		self.n_steps += 1

		_dist = self._get_dist(self.state, self.goal)
		self._update_position(action)

#		if self.dynamic_flag == 1:
#			dist = self._get_dist(self.dynamic_state, self.goal)
#			self.dynamic_flag = 0
#			message = "derror"
#		else:
		dist = self._get_dist(self.state, self.goal)

		if dist <= (self.dsize-1)*math.sqrt(2):
			reward = 1
			done = True
		elif self.n_steps == self.max_step:
			reward = 0
			done = True
		elif self.dynamic_flag == 1:
			reward = 0
			self.dynamic_flag = 0
			message = "derror"
#		elif dist < _dist:
#			reward = -0.1
		else:
			reward = -0.1


#		print(Actions(action))
#		print(self.map)

		obs = self._get_obs()

		return obs, reward, done, message

	def _get_dist(self, state1, state2):
		diff_x = state1[1] - state2[1]
		diff_y = state1[0] - state2[0]
		return math.sqrt(diff_x*diff_x + diff_y*diff_y)

	def _is_touching(self, dstate, obj):
		i = 0
		while True:
			j = 0
			while True:
				if self.map[dstate[1]+j][dstate[0]+i] == obj:
					return True
				j += 1
				if j == self.dsize:
					break
			i += 1
			if i == self.dsize:
				break

		return False

	def _update_position(self, action):
		state_ = list(self.state)

#		print(state_)
		if action == Actions.U:
			state_[1] -= 1
		elif action == Actions.R:
			state_[0] += 1
		elif action == Actions.D:
			state_[1] += 1
		elif action == Actions.L:
			state_[0] -= 1
		else:
			print("Unexpected action")
			return 0

		if (0 <= state_[1] <= self.h-self.dsize) and (0 <= state_[0] <= self.w-self.dsize) and\
		   (self._is_touching(state_, self.map_symbols.Dynamic_module) == False) and (self._is_touching(state_, self.map_symbols.Static_module) == False):
#			print("okok")
			i = 0
			while True:
				j = 0
				while True:
					self.map[self.state[1]+j][self.state[0]+i] = self.map_symbols.Health
					j += 1
					if j == self.dsize:
						break
				i += 1
				if i == self.dsize:
					break

#			print(Actions(action))
#			print(self.state)
			self.state = state_
#			print(self.state)

			# Set Droplet state
			i = 0
			while True:
				j = 0
				while True:
					self.map[self.state[1]+j][self.state[0]+i] = self.map_symbols.State
					j += 1
					if j == self.dsize:
						break
				i += 1
				if i == self.dsize:
					break

		elif (0 <= state_[1] < self.h-self.dsize+1) and (0 <= state_[0] < self.w-self.dsize+1) and\
		   (self._is_touching(state_, self.map_symbols.Dynamic_module) == True):
			self.dynamic_flag += 1
			self.dynamic_state = state_

			i = 0
			while True:
				j = 0
				while True:
					if self.map[state_[1]+j][state_[0]+i] == self.map_symbols.Dynamic_module:
						self.map[state_[1]+j][state_[0]+i] = self.map_symbols.Static_module
					j += 1
					if j == self.dsize:
						break
				i += 1
				if i == self.dsize:
					break

	def _get_obs(self):
		obs = np.zeros(shape = (3, self.w, self.h))
		for i in range(self.w):
			for j in range(self.h):
				if self.map[j][i] == self.map_symbols.State:
					obs[0][i][j] = 1
				elif self.map[j][i] == self.map_symbols.Goal:
					obs[1][i][j] = 1
				elif self.map[j][i] == self.map_symbols.Static_module:
					obs[2][i][j] = 1
#		print(obs)
		return obs

	def close(self):
		pass
