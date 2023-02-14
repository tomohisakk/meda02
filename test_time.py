import time
from env import MEDAEnv
import ppo
import torch as T

###### Set params ##########

w = 16
h = 16
s_modules = 12
d_modules = 12
N_EPOCH = 1

############################
save_name = str(w)+str(h)+"1"+str(s_modules)+str(d_modules)+ "/" + str(N_EPOCH)

env = MEDAEnv(w, h, 1, s_modules, d_modules, test_flag=False)

device = T.device('cpu')
net = ppo.PPO(env.observation_space, env.action_space).to(device)
net.load_checkpoint(save_name)


observation = env.reset()
observation = T.tensor([observation], dtype=T.float)

start_time = time.time()
output = net(observation)
forward_pass_time = time.time() - start_time

print("Forward pass time: ", forward_pass_time)