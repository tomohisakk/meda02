import os
import ptan
import time
import torch as T
import torch.optim as optim
import torch.nn.functional as F
from ignite.engine import Engine

from env import MEDAEnv
import common, ppo

import warnings
warnings.filterwarnings("ignore")

class Params():
###########################

	games = 10000
#	nepoches = 100
	lr = .001
	entropy_beta = .2
	batch_size = 256
	ppo_epoches = 10

	w = 16
	h = 16
	dsize = 1
	s_modules = 0
	d_modules = 12
	importf = None

#########################

	useGPU = True
	env_name = str(w)+str(h)+str(dsize)+str(s_modules)+str(d_modules)
	gamma = 0.99
	gae_lambda = 0.95
	ppo_eps =  0.2
	stop_test_reward = 10000
	stop_reward = None
#	n_actors = 4
	ppo_trajectory = 2560

params = Params()

if __name__ == "__main__":
	T.manual_seed(42)

#	envs = []
#	for _ in range(params.n_actors):
	env = MEDAEnv(w=params.w, h=params.h, dsize=params.dsize, s_modules=params.s_modules, d_modules=params.d_modules)
#		envs.append(env)

	if params.useGPU == True:
		device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
	else:
		device = T.device('cpu')
	print("Device is ", device)

	net = ppo.PPO(env.observation_space, env.action_space).to(device)
	if params.importf != None:
		net.load_checkpoint(params.importf)
	print(net)

	agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True,
								   preprocessor=ptan.agent.float32_preprocessor,
								   device=device)

	exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

#	optimizer = optim.Adam(net.parameters(), lr=params.lr, eps=0.1)
	optimizer = optim.SGD(net.parameters(), lr=params.lr, momentum=0.9)

	scheduler = T.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

	if not os.path.exists("saves"):
		os.makedirs("saves")
	if not os.path.exists("saves/" + params.env_name):
		os.makedirs("saves/" + params.env_name)

	def process_batch(engine, batch):
		start_ts = time.time()
		optimizer.zero_grad()
		res = {}

		states_t, actions_t, adv_t, ref_t, old_logprob_t = batch
		policy_t, value_t = net(states_t)
		loss_value_t = F.mse_loss(value_t.squeeze(-1), ref_t)
		res['ref'] = ref_t.mean().item()

		logpolicy_t = F.log_softmax(policy_t, dim=1)

		prob_t = F.softmax(policy_t, dim=1)
		loss_entropy_t = (prob_t * logpolicy_t).sum(dim=1).mean()

		logprob_t = logpolicy_t.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
		ratio_t = T.exp(logprob_t - old_logprob_t)
		surr_obj_t = adv_t * ratio_t
		clipped_surr_t = adv_t * T.clamp(ratio_t, 1.0 - params.ppo_eps, 1.0 + params.ppo_eps)
		loss_policy_t = T.min(surr_obj_t, clipped_surr_t).mean()

		loss_t = -loss_policy_t + loss_value_t + params.entropy_beta * loss_entropy_t
#		print(loss_t)
		loss_t.backward()
		optimizer.step()

		res.update({
			"loss": loss_t.item(),
			"loss_value": loss_value_t.item(),
			"loss_policy": loss_policy_t.item(),
			"loss_entropy": loss_entropy_t.item(),
		})

		return res

	engine = Engine(process_batch)

	common.setup_ignite(engine, params, exp_source, params.env_name, net, optimizer, scheduler ,extra_metrics=(
		'test_reward', 'avg_test_reward', 'test_steps'))

	engine.run(ppo.batch_generator(exp_source, net, params.ppo_trajectory,
									params.ppo_epoches, params.batch_size,
									params.gamma, params.gae_lambda, device=device))
