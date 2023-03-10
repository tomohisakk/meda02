import ptan
import pickle
import warnings
import torch as T
import collections
import numpy as np
import torch.nn as nn
from typing import Iterable
from ignite.engine import Engine
import ptan.ignite as ptan_ignite
from types import SimpleNamespace
from datetime import timedelta, datetime
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
import time
from PIL import Image, ImageOps

import ppo
from env import MEDAEnv
from map import MakeMap
from map import Symbols

import smtplib
from email.mime.text import MIMEText

def test(save_name, w, h, dsize, s_modules, d_modules):
	###### Set params ##########
	############################
	env = MEDAEnv(w, h, dsize, s_modules, d_modules, test_flag=True)

	device = T.device('cpu')
	net = ppo.PPO(env.observation_space, env.action_space).to(device)
	net.load_checkpoint(save_name)

	for param in net.parameters():
		param.requires_grad = False

	dir_name = "testmaps/%sx%s/%s/%s,%s"%(w , h, dsize, s_modules, d_modules)
	file_name = "%s/map.pkl"%(dir_name)

	save_file = open(file_name, "rb")
	maps = pickle.load(save_file)

#	print(net.actor[0].weight)
#	print(net.actor[0].bias)
#	print(net.actor[2].weight)
#	print(net.actor[2].bias)

	n_games = 0
	sum_total_derror = 0
	sum_total_step = 0
	n_critical = 0

	map_symbols = Symbols()
	mapclass = MakeMap(w,h,dsize,s_modules, d_modules)

	images = []

	for n_games in range(10):
		tmap = maps[n_games]

		observation = env.reset(test_map=tmap)

		img = env.render()
		img = Image.fromarray(img)
		images.append(img)

		done = False
		score = 0
		n_steps = 0
		n_degrad = 0

#		path = _compute_shortest_route(w, h, dsize, map_symbols, tmap, (0,0))

		while not done:
			observation = T.tensor([observation], dtype=T.float)
#			print(observation)
			with T.no_grad():
				net.eval()
				acts, _ = net(observation)
			action = T.argmax(acts).item()
			observation, reward, done, info = env.step(action)

			img = env.render()
			img = Image.fromarray(img)
			images.append(img)

			score += reward
			n_steps += 1

			sum_total_derror += info[0]

			if done:
				break

		sum_total_step += info[1]

		if env.max_step == n_steps:
			print(n_games)
			return 0

	images[0].save('animation.gif', format='GIF', append_images=images[1:], save_all=True, duration=500, loop=0)
	env.close()

	print("Avg critical path is", sum_total_step/100)
	print("Avg d_error is", sum_total_derror/100)

	save_file.close()

	return sum_total_step/100