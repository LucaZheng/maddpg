#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch as T
import torch.nn.functional as F

from maddpg_replaybuffer import *
from networks_agent import ActorNetwork, CriticNetwork, Agent


# In[10]:


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, agent_lists, save_dir, 
                 fc1_dims=64, fc2_dims=64, alpha=0.01, beta=0.01, tau=0.01, gamma=0.95):
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.alpha = alpha
        self.beta = beta
        self.agent_lists = agent_lists
        self.agents = {}
        for agent_idx, agent_name in enumerate(agent_lists):
            self.agents[agent_name] = Agent(actor_dims[agent_idx], critic_dims, n_agents, n_actions, 
                                     save_dir, agent_idx, alpha=alpha, beta=beta, 
                                     fc1_dims=fc1_dims, fc2_dims=fc2_dims, tau=tau, gamma=gamma)

    def save_agents(self):
        for agent in self.agents:
            self.agents[agent].save_models()
        print('agent models saved.')
        
    def load_agents(self):
        for agent in self.agents:
            self.agents[agent].load_models()
        print('agent models loaded.')
        
    def choose_action(self, agent, raw_obs):
        action = self.agents[agent].choose_action(raw_obs[agent])
        return action

    def train(self, buffer):
        # check if ready
        if not buffer.ready():
            return
            
        # forwardpass
        states, states_, actor_states, actor_states_, \
        actor_actions, rewards, dones = buffer.generate_batch()
        print(f"Training with: states.shape={states.shape}, actor_states[0].shape={actor_states[0].shape}")
        device = next(iter(self.agents.values())).actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        actor_actions = T.tensor(actor_actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        actor_new_actions = []
        actor_new_mu_actions = []
        actor_old_actions = []
        # store generated actions
        for idx, agent in enumerate(self.agent_lists):
            print(f"Agent {agent} - state shape: {actor_states[idx].shape}, action shape: {actor_actions[idx].shape}")
            # store actor action prediction
            actor_states = T.tensor(actor_states[idx], dtype=T.float).to(device)
            pi = self.agents[agent].actor.forward(actor_states)
            actor_new_mu_actions.append(pi)

            # store target actor action prediction
            actor_new_states = T.tensor(actor_states_[idx], dtype=T.float).to(device)
            new_pi = self.agents[agent].target_actor.forward(actor_new_states)
            actor_new_actions.append(new_pi)

            # store old action
            actor_old_actions.append(actor_actions[idx])

        # concat actions with tensor
        new_actions = T.cat([acts for acts in actor_new_actions], dim=1)
        mu_actions = T.cat([acts for acts in actor_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in actor_old_actions], dim=1)
        
        # loss
        for idx, agent in enumerate(self.agent_lists):
            # calculate critic loss
            critic_value_ = self.agents[agent].target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = self.agents[agent].critic.forward(states, old_actions).flatten()
            target = rewards[:, idx] + self.agents[agent].gamma * critic_value_

            critic_loss = F.mse_loss(target, critic_value)
            self.agents[agent].critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.agents[agent].critic.optimizer.step()

            # calculate actor loss
            actor_loss = self.agents[agent].actor.forward(states, mu_actions).flatten()
            actor_loss = -T.mean(actor_loss)
            self.agents[agent].actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.agents[agent].actor.optimizer.step()
            
            self.agents[agent].update_networks()


# In[ ]:




