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
from IPython.display import clear_output, display
import time

import ppo
from env import MEDAEnv
from map import MakeMap
from map import Symbols


import smtplib
from email.mime.text import MIMEText

def send_email(subject, body, sender, recipients, password):
	msg = MIMEText(body)
	msg['Subject'] = subject
	msg['From'] = sender
	msg['To'] = ', '.join(recipients)
	smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
	smtp_server.login(sender, password)
	smtp_server.sendmail(sender, recipients, msg.as_string())
	smtp_server.quit()


def setup_ignite(engine: Engine, params: SimpleNamespace, exp_source, run_name: str, 
				 net, optimizer, scheduler, extra_metrics: Iterable[str] = ()):
	warnings.simplefilter("ignore", category=UserWarning)
	handler = ptan_ignite.EndOfEpisodeHandler(exp_source, bound_avg_reward=np.inf)
	handler.attach(engine)
	ptan_ignite.EpisodeFPSHandler().attach(engine)

	total_rewards = []
	total_n_steps_ep = []

	good_results = []

	@engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
	def episode_completed(trainer: Engine):
		total_rewards.append(trainer.state.episode_reward)
		total_n_steps_ep.append(trainer.state.episode_steps)
		tmp = int(params.games/10)

		if trainer.state.episode % tmp == 0:
			mean_reward = np.mean(total_rewards[-tmp:])
			mean_n_steps = np.mean(total_n_steps_ep[-tmp:])
			passed = trainer.state.metrics.get('time_passed', 0)
			print("%d/%d: reward=%.2f, steps=%d, elapsed=%s" % (
				trainer.state.episode/params.games, trainer.state.episode, 
				mean_reward, mean_n_steps,
				timedelta(seconds=int(passed))))

		if trainer.state.episode%params.games == 0:
			save_name = params.env_name + "/" +str(int(trainer.state.episode/params.games))
			net.save_checkpoint(save_name)
			tmp = test(save_name, params.w, params.h, params.dsize, params.s_modules, params.d_modules)
			if tmp != 0:
				subject = "Learning end from GPU"
				body = "Successfully end"
				sender = "tomohisa.kawakami@tomiyama-lab.org"
				recipients = ["tomohisaee@gmail.com"]
				password = "22901900284"
				send_email(subject, body, sender, recipients, password)
				engine.terminate()
#				good_results.append(trainer.state.episode/params.games)
#				good_results.append(tmp)

#			if len(good_results)!=0 and trainer.state.episode/params.games == good_results[0]:
#				scheduler.step()
#				print("LR: ", optimizer.param_groups[0]['lr'])

#			if len(good_results)!=0 and trainer.state.episode/params.games == (good_results[0]+3):
#				print("=== Good Results===")
#				print(good_results)
#				subject = "Learning end from GPU"
#				body = "Successfully end"
#				sender = "tomohisa.kawakami@tomiyama-lab.org"
#				recipients = ["tomohisaee@gmail.com"]
#				password = "22901900284"
#				send_email(subject, body, sender, recipients, password)
#				engine.terminate()

	logdir = f"runs/{params.env_name}"
	tb = tb_logger.TensorboardLogger(log_dir=logdir)
	run_avg = RunningAverage(output_transform=lambda v: v['loss'])
	run_avg.attach(engine, "avg_loss")

	metrics = ['reward', 'steps', 'avg_reward']
	handler = tb_logger.OutputHandler(
		tag="episodes", metric_names=metrics)
	event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
	tb.attach(engine, log_handler=handler, event_name=event)

	ptan_ignite.PeriodicEvents().attach(engine)
	metrics = ['avg_loss']
	metrics.extend(extra_metrics)
	handler = tb_logger.OutputHandler(
		tag="train", metric_names=metrics,
		output_transform=lambda a: a)
	event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
	tb.attach(engine, log_handler=handler, event_name=event)


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

	map_symbols = Symbols()
	mapclass = MakeMap(w,h,dsize,s_modules, d_modules)

	n_critical = 0
	for n_games in range(100):
		tmap = maps[n_games]

		observation = env.reset(test_map=tmap)
		img = env.render()

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

			clear_output(wait=True)
			display(img)
			time.sleep(0.1)

			score += reward
			n_steps += 1

			sum_total_derror += info[0]

			if done:
				break

		sum_total_step += info[1]

		if env.max_step == n_steps:
			print(n_games)
			return 0
	
	env.close()

	print("Avg critical path is", sum_total_step/100)
	print("Avg d_error is", sum_total_derror/100)

	save_file.close()

	return sum_total_step/100