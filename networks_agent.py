#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


# In[2]:


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_agents, n_actions, fc1_dims, fc2_dims, beta, save_dir, save_name):
        super(CriticNetwork, self).__init__()
        self.file_name = os.path.join(save_dir, save_name)
        self.input_dims = input_dims
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.beta = beta
    
        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
    
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state,action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_name)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.file_name))

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, alpha, save_dir, save_name):
        super(ActorNetwork, self).__init__()

        self.file_name = os.path.join(save_dir, save_name)
        
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.alpha = alpha

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_name)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.file_name))


# In[1]:


class Agent:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, save_dir, agent_idx, alpha=0.01, beta=0.01, fc1_dims=64, fc2_dims=64, tau=0.01, gamma=0.95):
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.gamma = gamma
        self.agent_name = 'agent_%s' % agent_idx

        self.actor = ActorNetwork(actor_dims, fc1_dims, fc2_dims, n_actions, alpha, save_dir, save_name=self.agent_name+'actor')
        self.critic = CriticNetwork(critic_dims, n_agents, n_actions, fc1_dims, fc2_dims, beta, save_dir, save_name=self.agent_name+'critic')
        self.target_actor = ActorNetwork(actor_dims, fc1_dims, fc2_dims, n_actions, alpha, save_dir, save_name=self.agent_name+'target_actor')
        self.target_critic = CriticNetwork(critic_dims, n_agents, n_actions, fc1_dims, fc2_dims, beta, save_dir, save_name=self.agent_name+'target_critic')

        self.update_networks(tau=1)

    def choose_action(self, observation):
        obs = T.tensor([observation], dtype=T.float).to(self.actor.device)
        action = self.actor.forward(obs)
        #action = (action + 1) / 2
        noise = T.rand(self.n_actions).to(self.actor.device) * 0.1
        action = action + noise
        action = T.clamp(action,0,1)
    
        return action.detach().cpu().numpy()[0]
        
    def update_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        # update actor netwroks
        actor_params = self.actor.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        actor_dic = dict(actor_params)
        target_actor_dic = dict(target_actor_params)

        for name in actor_dic:
            actor_dic[name] = tau * actor_dic[name].clone() + (1-tau) * target_actor_dic[name].clone()

        self.target_actor.load_state_dict(actor_dic)

        # update critic networks
        critic_params = self.critic.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        critic_dic = dict(critic_params)
        target_critic_dic = dict(target_critic_params)

        for name in critic_dic:
            critic_dic[name] = tau * critic_dic[name].clone() + (1-tau) * target_critic_dic[name].clone()

        self.target_critic.load_state_dict(critic_dic)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

