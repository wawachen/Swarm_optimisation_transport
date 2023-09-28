import matplotlib.pyplot as plt
import numpy as np

import torch
# from pyrep.policies.model_attention import Actor_att
from torch.utils.tensorboard import SummaryWriter

print("wawa")
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter(log_dir='./log')
net = Actor_att(32, n_good=3, n_food = 3, num_units=32, with_action = False)
obs = torch.tensor([0.2,0.2,0.3,0.3,0.1,0.1,0.2,0.2,0.6,0.6,0.8,0.8])
writer.add_graph(net, obs)
writer.close()