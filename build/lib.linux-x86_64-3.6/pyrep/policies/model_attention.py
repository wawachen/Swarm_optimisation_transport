import torch 
import math 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 32 ouput action

class Actor_att(nn.Module):
    def __init__(self, num_outputs, n_good=3, n_food = 3, num_units=32, with_action = False):
        super(Actor_att,self).__init__()
        self.num_test = num_units // 2
        self.num_units = num_units
        self.n_good = n_good
        self.n_food = n_food
        self.with_action = with_action

        self.layer_self1 = nn.Linear(2, num_units) 
        self.layer_self2 = nn.Linear(num_units, self.num_test)
        self.layer_food1 = nn.Linear(2, num_units)
        self.layer_food2 = nn.Linear(num_units, self.num_test)
        self.food_norm = nn.LayerNorm(self.num_test)
        self.layer_other1 = nn.Linear(2,num_units)
        self.layer_other2 = nn.Linear(num_units,self.num_test)
        self.other_norm = nn.LayerNorm(self.num_test)
        self.layer_merge1 = nn.Linear(self.num_test*3,num_units)
        self.layer_merge2 = nn.Linear(num_units,num_units)
        self.layer_merge3 = nn.Linear(num_units,num_outputs)

        if with_action:
            self.layer_self1 = nn.Linear(4,num_units)
            

    def forward(self, input):
        #self mlp
        # print(input.shape)
        self_in = input[:,:2]
        # print(self_in.shape)
        if self.with_action:
            #obs_self+ food+other_obs+self_action
            self_action = input[:,-2:]
            self_in = torch.cat([self_in, self_action], dim=1)

        # print("self",self_in.shape)
        self_out = F.relu(self.layer_self1(self_in))
        self_out = F.relu(self.layer_self2(self_out))
        
        #food mlp
        if self.with_action:
            food_input = input[:,2:2+self.n_food*2]
        else:
            food_input = input[:,2:2+self.n_food*2]

        food_input = torch.split(food_input, 2, dim=1)
        # print(len(food_input))
        food_input = torch.stack(list(food_input), dim=0)
        
        fc1_out = F.relu(self.layer_food1(food_input))
        food_outs = F.relu(self.layer_food2(fc1_out))

        food_out = food_outs.permute(1, 2, 0)
        # print(torch.matmul(self_out.unsqueeze(1), food_out).shape)
        food_out_attn = F.softmax(torch.matmul(self_out.unsqueeze(1), food_out)/math.sqrt(self.num_test),dim=2)
        food_out = torch.matmul(food_out_attn, food_out.permute(0,2,1)).squeeze(1) 
        food_out = self.food_norm(food_out)
        food_out = F.relu(food_out)
        # print(food_out.shape)
        
        #other mlp
        if self.with_action:
            other_good_in = input[:, 2+self.n_food*2:2+self.n_food*2+(self.n_good-1)*2]
        else:
            other_good_in = input[:, 2+self.n_food*2:2+self.n_food*2+(self.n_good-1)*2]
        other_good_ins = []
        for i in range(self.n_good-1):
            pos = other_good_in[:, i*2:(i+1)*2]
            other_good_ins.append(pos)

        other_good_ins = torch.stack(other_good_ins, dim=0)
        # print(other_good_ins.shape)
        fc1_other = F.relu(self.layer_other1(other_good_ins))
        other_outs = F.relu(self.layer_other2(fc1_other))
        other_good_out = other_outs.permute(1, 2, 0) 

        other_good_out_attn = F.softmax(torch.matmul(self_out.unsqueeze(1), other_good_out)/math.sqrt(self.num_test),dim=2)
        # print("attn:", other_good_out_attn.shape)
        other_good_out = torch.matmul(other_good_out_attn, other_good_out.permute(0,2,1)).squeeze(1)
        other_good_out = self.other_norm(other_good_out)
        other_good_out = F.relu(other_good_out)
        # print(other_good_out.shape)

        input_merge = torch.cat([self_out, food_out, other_good_out], 1)
        # print(input_merge.shape)

        if self.with_action:
            out = F.relu(self.layer_merge1(input_merge))
            out = F.relu(self.layer_merge2(out))
        else:
            out = F.leaky_relu(self.layer_merge1(input_merge))
            # print(out)
            out = F.leaky_relu(self.layer_merge2(out))

        out = self.layer_merge3(out)

        return out


class Critic_att(nn.Module):
    def __init__(self, num_outputs, index, n_good=3, n_food=3, num_units=64):
        super(Critic_att,self).__init__()
        self.num_test = num_units // 2
        self.n_good = n_good
        self.n_food = n_food
        self.num_outputs = num_outputs
        self.num_units = num_units
        self.index = index
        self.layer_actor1 = nn.Linear(self.num_test, self.num_test)
        self.layer_actor2 = nn.Linear(self.num_test, self.num_test)
        self.layer_actor3 = nn.Linear(self.num_test, self.num_test)
        self.layer_merge1 = nn.Linear(num_units,num_units)
        self.layer_merge2 = nn.Linear(num_units,num_units)
        self.layer_merge3 = nn.Linear(num_units,num_outputs)
        self.critic_norms = nn.ModuleList([nn.LayerNorm(self.num_test) for _ in range(self.n_good)])
        self.critic_norm2 = nn.LayerNorm(self.num_test)


    def forward(self,input):
        input_action = input[:, -2*(self.n_good):]
        self_action = input_action[:, self.index*2: (self.index+1)*2]
        good_action = input_action[:, :]

        # split self obs
        length_obs = 2+self.n_food*2+(self.n_good-1)*2
        self_start = (self.index)*length_obs

        # self mlp
        input_obs_self = input[:, self_start:self_start+length_obs]
        self_in = input_obs_self
        self_in = torch.cat([self_in, self_action], 1)
        actor_model = Actor_att(self.num_test, n_good=self.n_good, n_food=self.n_food, num_units=self.num_units, with_action=True) 

        # print(self_in.shape)
        self_out = actor_model.forward(self_in)

        # other agent mlp
        other_good_ins = []
        for i in range(self.n_good):
            if i==self.index:
                continue
            other_good_beg = i*length_obs
            other_good_in = input[:,other_good_beg:other_good_beg+length_obs]
            tmp = torch.cat([other_good_in, good_action[:, i*2:(i+1)*2]], 1)
            # print(tmp.shape)
            other_good_ins.append(tmp)
        other_good_outs = []

        batch_other_good_ins = torch.stack(other_good_ins, dim=0)
        actor_models = [Actor_att(self.num_test,
                                n_good=self.n_good, n_food=self.n_food, num_units=self.num_units,
                                with_action=True) for _ in range(self.n_good-1)]
        # print(batch_other_good_ins[0,:,:].shape)
        other_good_outs = [model.forward(batch_other_good_ins[i,:,:]) for i,model in enumerate(actor_models)]

        theta_out = []
        phi_out = []
        g_out = []

        theta_out.append(self_out)
        phi_out.append(self_out)
        g_out.append(self_out)

        for i, out in enumerate(other_good_outs):
            theta_out.append(out)
            phi_out.append(out)
            g_out.append(out)

        theta_outs = torch.stack(theta_out, 0)
        theta_outs = F.relu(self.layer_actor1(theta_outs))
        theta_outs = theta_outs.permute(1,2,0)
        # print(theta_outs.shape)
        phi_outs = torch.stack(phi_out, 0)
        phi_outs = F.relu(self.layer_actor2(phi_outs))
        phi_outs = phi_outs.permute(1,2,0)

        g_outs = torch.stack(g_out, 0)
        g_outs = F.relu(self.layer_actor3(g_outs))
        g_outs = g_outs.permute(1,2,0)
        
        # print(torch.matmul(theta_outs, phi_outs.permute(0,2,1)).shape)
        self_attention = F.softmax(torch.matmul(theta_outs, phi_outs.permute(0,2,1))/math.sqrt(self.num_test),dim=2)
        # print(g_outs.shape,'self_attention')
        input_all = torch.matmul(self_attention, g_outs)
        input_all_new = []
        
        for i in range(self.n_good):
            critic_norm = self.critic_norms[i]
            # print(input_all[:,:,i].shape)
            input_all_new.append(critic_norm(input_all[:,:,i]))
            # print(input_all[:,:,i].shape)
        
        input_all = torch.stack(input_all_new, 2)
        # print(input_all.shape)
        # input_all = tf.contrib.layers.layer_norm(input_all)
        input_all = F.relu(input_all)

        self_out_new = input_all[:,:,0]
        good_out_new = input_all[:,:,1:self.n_good]

        other_good_out_attn = F.softmax(torch.matmul(self_out_new.unsqueeze(1), good_out_new)/math.sqrt(self.num_test),dim=2)
        other_good_out = torch.matmul(other_good_out_attn, good_out_new.permute(0,2,1)).squeeze(1)
        # print(other_good_out.shape)
        other_good_out = self.critic_norm2(other_good_out)
        other_good_out = F.relu(other_good_out)
        # print(other_good_out.shape)

        # merge layer for all
        input_merge = torch.cat([self_out, other_good_out], 1)
        # print(input_merge.shape)

        out = F.leaky_relu(self.layer_merge1(input_merge))
        out = F.leaky_relu(self.layer_merge2(out))
        out = self.layer_merge3(out)

        print(out.shape)

        return out

