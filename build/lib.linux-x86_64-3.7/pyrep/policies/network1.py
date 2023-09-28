import torch
import torch.nn as nn
import torch.nn.functional as f
from tensorboardX import SummaryWriter
from torch import tanh

# for initalising all layers of a network at once
def initialize_weights(net, low=-3e-2, high=3e-2):
    for param in net.parameters():
        param.data.uniform_(low, high)

# the actor takes a state and outputs an estimated best action
class Actor(nn.Module):
    def __init__(self, action_size, state_size, hidden_in_size, hidden_out_size, action_type):
        super(Actor, self).__init__()

        self.action_type = action_type
        self.fc1 = nn.Linear(state_size,hidden_in_size)
        self.fc2 = nn.Linear(hidden_in_size,hidden_out_size)
        self.fc3 = nn.Linear(hidden_out_size,action_size)
        initialize_weights(self)


    def forward(self, state):
        layer_1 = f.relu(self.fc1(state))
        layer_2 = f.relu(self.fc2(layer_1))
        if self.action_type=='continuous':
            action = torch.tanh(self.fc3(layer_2)) # tanh because the action space is -1 to 1
        elif self.action_type=='discrete':
            action = f.softmax(self.fc3(layer_2), dim=-1) # softmax because the action space is discrete
        return action

# with SummaryWriter(comment='LinearInLinear') as w:
#     w.add_graph(Actor())

# the critic takes the states and actions of both agents and outputs a prob distribution over estimated Q-values
class Critic(nn.Module):
    def __init__(self, actions_size, states_size, hidden_in_size, hidden_out_size, num_atoms): # num_atoms is the granularity of the bins
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(states_size,hidden_in_size)
        self.fc2 = nn.Linear(hidden_in_size+actions_size,hidden_out_size) # the actions are added to the second layer
        self.fc3 = nn.Linear(hidden_out_size,num_atoms)
        initialize_weights(self)


    def forward(self, states, actions, log=False): # log determines whether the softmax or log softmax is outputed for critic updates
        layer_1 = f.relu(self.fc1(states))
        layer_1_cat = torch.cat([layer_1, actions], dim=1)
        layer_2 = f.relu(self.fc2(layer_1_cat))
        Q_probs = self.fc3(layer_2)
        if log:
            return f.log_softmax(Q_probs, dim=-1)
        else:
            return f.softmax(Q_probs, dim=-1) # softmax converts the Q_probs to valid probabilities (i.e. 0 to 1 and all sum to 1)

class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, outputs)

    def forward(self, x):
        x = tanh(self.fc1(x))
        x = f.relu(self.fc2(x))
        return self.fc3(x)

class DQN_conv(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN_conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))