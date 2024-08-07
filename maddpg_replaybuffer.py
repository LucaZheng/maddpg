"""
parameters:
max_size
actor_dims
critic_dims
n_actions
n_agents
batch_size

methods:
remember(self, raw_obs, state, action, reward, raw_obs_, state_, done)
generate_batch(self):return states, states_, actor_states, actor_states_, actor_actions, rewards, dones
def ready(self):return True if self.mem_count >= self.batch_size:
"""


import numpy as np

class MADDPG_Replaybuffer:
   
    def __init__(self, max_size, actor_dims, critic_dims, n_actions, n_agents, batch_size):
        self.max_mem = max_size
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.mem_count = 0
        
        self.state_memory = np.zeros((self.max_mem, self.critic_dims))
        self.new_state_memory = np.zeros((self.max_mem, self.critic_dims))
        self.reward_memory = np.zeros((self.max_mem, self.n_agents))
        self.terminal_memory = np.zeros((self.max_mem, self.n_agents), dtype=bool)
        
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []
        for i in range(self.n_agents):
            self.actor_state_memory.append(np.zeros((self.max_mem, self.actor_dims[i])))
            self.actor_new_state_memory.append(np.zeros((self.max_mem, self.actor_dims[i])))
            self.actor_action_memory.append(np.zeros((self.max_mem, self.n_actions)))

    def remember(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        index = self.mem_count % self.max_mem

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]
            
        self.mem_count += 1
        print(f"Remember: index={index}, mem_count={self.mem_count}")

    def generate_batch(self):
        max_sample = min(self.max_mem, self.mem_count)
        batch = np.random.choice(max_sample, self.batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        actor_states = []
        actor_states_ = []
        actor_actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_states_.append(self.actor_new_state_memory[agent_idx][batch])
            actor_actions.append(self.actor_action_memory[agent_idx][batch])

        print(f"Batch generated: states.shape={states.shape}, actor_states[0].shape={actor_states[0].shape}")
        return states, states_, actor_states, actor_states_, actor_actions, rewards, dones
        
    def ready(self):
        if self.mem_count >= self.batch_size:
            return True

