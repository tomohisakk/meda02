import gym
import math
import random
import numpy as np
from enum import IntEnum
from PIL import ImageDraw, Image

from map import MakeMap
from map import Symbols

class Actions(IntEnum):
	U = 0
	R = 1
	D = 2
	L = 3

	# diagnal move
	UR = 4
	UL = 5
	DR = 6
	DL = 7

	# change shape
	C1 = 8
	C2 = 9
	C3 = 10
	C4 = 11

class MEDAEnv(gym.Env):
	def __init__(self, w=8, h=8, dsize=1, s_modules=2, d_modules=2, test_flag=False):
		super(MEDAEnv, self).__init__()
		assert w>0 and h>0 and dsize>0
		assert 0<=s_modules and 0<=d_modules
		self.w = w
		self.h = h
		self.dsize = dsize
		self.actions = Actions
		self.action_space = len(self.actions)
		self.observation_space = (w, h, 3)
		self.n_steps = 0
		self.max_step = 2*(w+h)

		self.state = [0,0]
		self.goal = (w-1, h-1)

		self.map_symbols = Symbols()
		self.mapclass = MakeMap(w=self.w,h=self.h,dsize=self.dsize,s_modules=s_modules,d_modules=d_modules)
		self.map = self.mapclass.gen_random_map()

		self.test_flag = test_flag

		self.dynamic_flag = 0
#		self.dynamic_state = (0,0)

		self.is_vlong = False

		self.is_move = False

		self.total_step = 0
		self.initial_map = None


	def reset(self, test_map=None):
		self.n_steps = 0
		self.state = [0, 0]
		self.is_move = False
		self.dynamic_flag = 0

		if self.test_flag == False:
			self.map = self.mapclass.gen_random_map()
		else:
			self.map = test_map
		
		self.initial_map = self.map.copy()

		obs = self._get_obs()
		self.is_vlong = False

		self.total_step = 0

		return obs

	def step(self, action):
		self.is_move = False
		done = False
		message = None
		self.n_steps += 1
		info = [0,0]

		self._update_position(action)


		if self.is_vlong and self.state[0]==self.w-1 and self.state[1]==self.h-2:
			reward = 0
			done = True
#			print("hgoal")
		elif self.is_vlong==False and self.state[0]==self.w-2 and self.state[1]==self.h-1:
			reward = 0
			done = True
#			print("vgoal")
		elif self.n_steps == self.max_step:
			reward = -1.0
			done = True
		elif self.dynamic_flag == 1:
			if self.is_vlong:
				if Actions.UR <= action <= Actions.DL:
					reward = -0.4
				elif action == Actions.R or action == Actions.L:
					reward = -0.2
				else:
					reward = -0.1
			else:
				if Actions.UR <= action <= Actions.DL:
					reward = -0.4
				elif action == Actions.U or action == Actions.D:
					reward = -0.2
				else:
					reward = -0.1
			self.dynamic_flag = 0
			#message = "derror"
			info[0] = 1
#		elif self.is_move:
		elif self.is_vlong:
			if Actions.UR <= action <= Actions.DL:
				reward = -0.4
			elif action == Actions.R or action == Actions.L:
				reward = -0.2
			else:
				reward = -0.1
		else:
			if Actions.UR <= action <= Actions.DL:
				reward = -0.4
			elif action == Actions.U or action == Actions.D:
				reward = -0.2
			else:
				reward = -0.1

		obs = self._get_obs()

		if self.is_move:
			if self.is_vlong:
				if Actions.UR <= action <= Actions.DL:
					self.total_step += 4
				elif action == Actions.R or action == Actions.L:
					self.total_step += 2
				else:
					self.total_step += 1
			else:
				if Actions.UR <= action <= Actions.DL:
					self.total_step += 4
				elif action == Actions.U or action == Actions.D:
					self.total_step += 2
				else:
					self.total_step += 1

		info[1] = self.total_step

#		print(self.map)

		return obs, reward, done, info

#	def _get_dist(self, state1, state2):
#		diff_x = state1[1] - state2[1]
#		diff_y = state1[0] - state2[0]
#		return math.sqrt(diff_x*diff_x + diff_y*diff_y)

	def _is_touching(self, state, obj):
		if self.is_vlong:
			if self.map[state[1]][state[0]] == obj or self.map[state[1]+1][state[0]] == obj:
				return True
			else:
				return False
		else:
			if self.map[state[1]][state[0]] == obj or self.map[state[1]][state[0]+1] == obj:
				return True
			else:
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

		# diagnal movements
		elif action == Actions.UR:
			state_[0] += 1
			state_[1] -= 1
		elif action == Actions.UL:
			state_[1] -= 1
			state_[0] -= 1
		elif action == Actions.DR:
			state_[1] += 1
			state_[0] += 1
		elif action == Actions.DL:
			state_[1] += 1
			state_[0] -= 1

		elif action == Actions.C1:
			if self.is_vlong:
				#L
				state_[0] -= 1
				self.is_vlong = False
			else:
				#U
				state_[1] -= 1
				self.is_vlong = True
		elif action == Actions.C2:
			if self.is_vlong:
				#DL
				state_[1] += 1
				state_[0] -= 1
				self.is_vlong = False
			else:
				#UR
				state_[0] += 1
				state_[1] -= 1
				self.is_vlong = True
		elif action == Actions.C3:
			if self.is_vlong:
				#same
				self.is_vlong = False
			else:
				#same
				self.is_vlong = True
		elif action == Actions.C4:
			if self.is_vlong:
				#D
				state_[1] += 1
				self.is_vlong = False
			else:
				#R
				state_[0] += 1
				self.is_vlong = True
		else:
			print("Unexpected action")
			return 0

		if self.is_vlong:
			if 0>state_[0] or 0>state_[1] or state_[0]>self.w-1 or state_[1]+1>self.h-1:
				if 8 <= action <= 11:
					self.is_vlong = False
				return

			if 4<=action<=7:
				if self.map[self.state[1]][state_[0]]== self.map_symbols.Dynamic_module:
					self.map[self.state[1]][state_[0]] = self.map_symbols.Static_module
				if self.map[state_[1]][self.state[0]]== self.map_symbols.Dynamic_module:
					self.map[state_[1]][self.state[0]] = self.map_symbols.Static_module
				if self.map[self.state[1]+1][state_[0]]== self.map_symbols.Dynamic_module:
					self.map[self.state[1]+1][state_[0]] = self.map_symbols.Static_module
				if self.map[state_[1]+1][self.state[0]]== self.map_symbols.Dynamic_module:
					self.map[state_[1]+1][self.state[0]] = self.map_symbols.Static_module
				if self.map[self.state[1]][state_[0]]==self.map_symbols.Static_module or\
				   self.map[state_[1]][self.state[0]]==self.map_symbols.Static_module or\
				   self.map[self.state[1]+1][state_[0]]==self.map_symbols.Static_module or\
				   self.map[state_[1]+1][self.state[0]]==self.map_symbols.Static_module:
						return

			if self._is_touching(state_, self.map_symbols.Dynamic_module):
				self.dynamic_flag = 1
				if self.map[state_[1]][state_[0]] == self.map_symbols.Dynamic_module:
					self.map[state_[1]][state_[0]] = self.map_symbols.Static_module
				if self.map[state_[1]+1][state_[0]] == self.map_symbols.Dynamic_module:
					self.map[state_[1]+1][state_[0]] = self.map_symbols.Static_module
				if 8 <= action <= 11:
					self.is_vlong = False
				return
			
			if self._is_touching(state_, self.map_symbols.Static_module):
				if 8 <= action <= 11:
					self.is_vlong = False
				return

			if 8 <= action <= 11:
				self.map[self.state[1]][self.state[0]] = self.map_symbols.Health
				self.map[self.state[1]][self.state[0]+1] = self.map_symbols.Health
			else:
				self.map[self.state[1]][self.state[0]] = self.map_symbols.Health
				self.map[self.state[1]+1][self.state[0]] = self.map_symbols.Health
			self.state = state_
			self.is_move = True
			self.map[self.state[1]][self.state[0]] = self.map_symbols.State
			self.map[self.state[1]+1][self.state[0]] = self.map_symbols.State

		else:
			if 0>state_[0] or 0>state_[1] or state_[0]+1>self.w-1 or state_[1]>self.h-1:
				if 8 <= action <= 11:
					self.is_vlong = True
				return

			if 4<=action<=7:
				if self.map[self.state[1]][state_[0]]== self.map_symbols.Dynamic_module:
					self.dynamic_flag = 1
					self.map[self.state[1]][state_[0]] = self.map_symbols.Static_module
				if self.map[state_[1]][self.state[0]]== self.map_symbols.Dynamic_module:
					self.dynamic_flag = 1
					self.map[state_[1]][self.state[0]] = self.map_symbols.Static_module
				if self.map[self.state[1]][state_[0]+1]== self.map_symbols.Dynamic_module:
					self.dynamic_flag = 1
					self.map[self.state[1]][state_[0]+1] = self.map_symbols.Static_module
				if self.map[state_[1]][self.state[0]+1]== self.map_symbols.Dynamic_module:
					self.dynamic_flag = 1
					self.map[state_[1]][self.state[0]+1] = self.map_symbols.Static_module
				if self.map[self.state[1]][state_[0]]==self.map_symbols.Static_module or\
				   self.map[state_[1]][self.state[0]]==self.map_symbols.Static_module or\
				   self.map[self.state[1]][state_[0]+1]==self.map_symbols.Static_module or\
				   self.map[state_[1]][self.state[0]+1]==self.map_symbols.Static_module:
						return

			if self._is_touching(state_, self.map_symbols.Dynamic_module):
				self.dynamic_flag = 1
				if self.map[state_[1]][state_[0]] == self.map_symbols.Dynamic_module:
					self.map[state_[1]][state_[0]] = self.map_symbols.Static_module
				elif self.map[state_[1]][state_[0]+1] == self.map_symbols.Dynamic_module:
					self.map[state_[1]][state_[0]+1] = self.map_symbols.Static_module
				if 8 <= action <= 11:
					self.is_vlong = True
				return
			
			if self._is_touching(state_, self.map_symbols.Static_module):
				if 8 <= action <= 11:
					self.is_vlong = True
				return

			if 8 <= action <= 11:
				self.map[self.state[1]][self.state[0]] = self.map_symbols.Health
				self.map[self.state[1]+1][self.state[0]] = self.map_symbols.Health
			else:
				self.map[self.state[1]][self.state[0]] = self.map_symbols.Health
				self.map[self.state[1]][self.state[0]+1] = self.map_symbols.Health
			self.state = state_
			self.is_move = True
			self.map[self.state[1]][self.state[0]] = self.map_symbols.State
			self.map[self.state[1]][self.state[0]+1] = self.map_symbols.State

	def _get_obs(self):
		obs = np.zeros(shape = (self.w, self.h, 3))
		for i in range(self.w):
			for j in range(self.h):
				if self.map[j][i] == self.map_symbols.State:
					obs[i][j][0] = 1
				elif self.map[j][i] == self.map_symbols.Goal:
					obs[i][j][1] = 1
				elif self.map[j][i] == self.map_symbols.Static_module:
					obs[i][j][2] = 1
#		print(obs)
		return obs

	def render(self, cell_size=30, border_width=1, bg_color=(255, 255, 255)):
		img01 = self._get_obs().astype(np.uint8)
		img_size = (self.w*cell_size, self.h*cell_size)
		img = Image.new('RGB', img_size, color=bg_color)
		draw = ImageDraw.Draw(img)

		for x in range(self.w):
			for y in range(self.h):
				cell_left = x*cell_size
				cell_upper = y*cell_size
				cell_right = (x+1)*cell_size
				cell_lower = (y+1)*cell_size

				if img01[x][y][0] == 1:
					draw.rectangle((cell_left, cell_upper, cell_right, cell_lower), fill=(0, 0, 255), outline=(0, 0, 0), width=border_width)
				elif img01[x][y][1] == 1:
					draw.rectangle((cell_left, cell_upper, cell_right, cell_lower), fill=(255, 0, 0), outline=(0, 0, 0), width=border_width)
				elif img01[x][y][2] == 1 and self.initial_map[y][x] == self.map_symbols.Static_module:
					draw.rectangle((cell_left, cell_upper, cell_right, cell_lower), fill=(192, 192, 192), outline=(0, 0, 0), width=border_width)
				elif img01[x][y][2] == 1 and self.initial_map[y][x] == self.map_symbols.Dynamic_module:
					draw.rectangle((cell_left, cell_upper, cell_right, cell_lower), fill=(0, 255, 0), outline=(0, 0, 0), width=border_width)
				else:
					draw.rectangle((cell_left, cell_upper, cell_right, cell_lower), fill=bg_color, outline=(0, 0, 0), width=border_width)

		return np.array(img)

	def close(self):
		pass
