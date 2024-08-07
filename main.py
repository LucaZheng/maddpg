# -*- coding: utf-8 -*-
"""MADDPG-simpleadversary.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XejrOl_5nShK_CN5zCTl-hEvDLV81e-C
"""

# install pettingzoo
# import maddpg, networks, replaybuffer from directory
import numpy as np
from pettingzoo.mpe import simple_adversary_v3

"""# Helper Function: actor states -> global state"""

def obs_list_to_state_vector(observation):
    obs_array = [np.array(obs) for obs in observation]
    return np.hstack(obs_array)

def prepare_transactions(obs, obs_, actions, rewards, truncations, agents):
    obs_list = []
    new_obs_list = []
    action_list = []
    reward_list = []
    dones_list = []
    for agent in agents:
        obs_list.append(obs[agent])
        new_obs_list.append(obs_[agent])
        action_list.append(actions[agent])
        reward_list.append(rewards[agent])
        dones_list.append(truncations[agent])
        #print(f'***preprocess on {agent} finished: obs shapes: {obs[agent].shape}, dones: {dones_list}***')

    states = obs_list_to_state_vector(obs_list)
    new_states = obs_list_to_state_vector(new_obs_list)

    return obs_list, new_obs_list, states, new_states, action_list, reward_list, dones_list

"""# Main"""

# hyper parameters
n_games = 50000
n_good_agents = 2
n_agents = n_good_agents + 1
max_steps = 50
total_steps = 0
score_history = []
best_score = 0
print_intervel = 5
evaluate = False

env = simple_adversary_v3.parallel_env(render_mode="rgb_array", N=n_good_agents, max_cycles=max_steps, continuous_actions=False)
observations, infos = env.reset()
actor_dims = []
for agent in env.agents:
    actor_dims.append(env.observation_space(agent).shape[0])
critic_dims = sum(actor_dims)
#n_actions = env.action_space(agent).shape[0]
n_actions = env.action_space(agent).n

# check shapes
print('agent list: ' + str(env.agents))
print('actor_dims: ' + str(actor_dims))
print('critic_dims: ' + str(critic_dims))
print('n_actions: ' + str(n_actions))

# initialize agents and memory
maddpg = MADDPG(actor_dims=actor_dims, critic_dims=critic_dims, n_agents=n_agents, agent_lists=env.agents, n_actions=n_actions, \
           save_dir='/content/agents')
memory = MADDPG_Replaybuffer(1000000, actor_dims=actor_dims, critic_dims=critic_dims,
                        n_actions=n_actions, n_agents=n_agents, batch_size=1024)#1024

for i in range(n_games):
    episode_steps = 0
    scores = 0
    obs, _ = env.reset()
    agents = env.agents.copy()
    while env.agents:
        episode_steps += 1
        total_steps += 1
        actions = {agent: maddpg.choose_action(agent, obs) for agent in env.agents}
        obs_, rewards, _, truncations, _ = env.step(actions)

        # store transactions based on current agents availability
        if env.agents:
            agents = env.agents.copy()  # update agents list

        # preprocess transactions
        obs_list, new_obs_list, state_list, new_state_list, action_list,\
        reward_list, dones_list = prepare_transactions(obs, obs_, actions, rewards, truncations, agents)

        # store transactions
        memory.remember(obs_list, state_list, action_list, reward_list, new_obs_list, new_state_list, dones_list)

        # training
        if total_steps % 50 == 0:
            train_counts = maddpg.train(memory)

        # update new obs
        obs = obs_
        scores += sum(reward_list)

    score_history.append(scores)
    avg_score = np.mean(score_history[-100:])
    print(f'-----------total_steps: {total_steps}, trained: {train_counts}, games: {i+1}------------\n')
    if not evaluate:
        if avg_score > best_score:
            maddpg.save_agents()
            best_score = avg_score
    if i % print_intervel == 0 and i > 0:
            print('episode', i+1, 'average score {:.1f}'.format(avg_score))

env.close()