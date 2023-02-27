from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_agents.trajectories
from catanatron import Player, Color
from catanatron.models.actions import generate_playable_actions
from catanatron.state_functions import player_key
from catanatron_gym.envs.catanatron_env import to_action_space, from_action_space
from catanatron_gym.features import create_sample, get_feature_ordering
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import trajectory, StepType
from tf_agents.utils import common


feature_ordering=get_feature_ordering(3)

def train_eval(
		actions_count=289 + 1,
		env_name="catanatron_gym:catanatron-v0",  # @param {type:"string"}
		num_iterations=2000,  # @param {type:"integer"}
		initial_collect_steps=300,  # @param {type:"integer"}
		collect_steps_per_iteration=10000,  # @param {type:"integer"}
		replay_buffer_capacity=100000,  # @param {type:"integer"}
		train_steps_per_iteration=10,
		fc_layer_params=(1024, 1024),
		episode_step_limit=9999,
		batch_size=64,  # @param {type:"integer"}
		learning_rate=1e-3,  # @param {type:"number"}
		gamma=0.99,
		log_interval=200,  # @param {type:"integer"}
		num_atoms=51,  # @param {type:"integer"}
		min_q_value=-1,  # @param {type:"integer"}
		max_q_value=1,  # @param {type:"integer"}
		n_step_update=2,  # @param {type:"integer"}
		num_eval_episodes=20,  # @param {type:"integer"}
		eval_interval=1000,  # @param {type:"integer"}
		discount=0.99,
	):
	pass

actions_count = 289 + 1

# env_name = "CartPole-v1" # @param {type:"string"}
env_name = "catanatron_gym:catanatron-v0"  # @param {type:"string"}
num_iterations = 2000  # @param {type:"integer"}

initial_collect_steps = 300  # @param {type:"integer"}
collect_steps_per_iteration = 10000  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}
train_steps_per_iteration = 10
fc_layer_params = (1024, 1024)
episode_step_limit = 9999
batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = 200  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = -1  # @param {type:"integer"}
max_q_value = 1  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}

num_eval_episodes = 20  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

feature_ordering = get_feature_ordering(3)
discount = 0.99


player_rewards = {
	Color.RED: 0,
	Color.WHITE: 0,
	Color.BLUE: 0
}

turns = 0


def my_reward_function(game, color):
	winning_color = game.winning_color()
	vps = game.state.player_state[f"{player_key(game.state, color)}_ACTUAL_VICTORY_POINTS"]
	vps -= player_rewards[color]
	player_rewards[color] = game.state.player_state[f"{player_key(game.state, color)}_ACTUAL_VICTORY_POINTS"]
	vps /= 2
	vps -= 0.005
	if color == winning_color:
		vps += 1
	return vps


tensorboard_base_dir = "C:\\Users\\manof\\PycharmProjects\\pytan_fast\\pytan_fast\\test\\tensorboard"
train_dir = os.path.join(tensorboard_base_dir, 'train')
eval_dir = os.path.join(tensorboard_base_dir, 'eval')



class FooPlayer(Player):
	def __init__(self, color, player_index):
		super().__init__(color)
		self.player_index = player_index
		self.env = None
		self.agent = None
		self.replay_buffer = None
		self.dataset = None
		self.iterator = None
		self.episode_rewards = 0
		self.train_step_counter = tf.Variable(0)
		self.agent_dir = os.path.join(tensorboard_base_dir, "agent{}".format(player_index))
		self.agent_train_dir = os.path.join(self.agent_dir, 'train')
		self.agent_eval_dir = os.path.join(self.agent_dir, 'eval')
		self._train_summary_writer = tf.compat.v2.summary.create_file_writer(self.agent_train_dir, flush_millis=10 * 1000)

	def make_agent(self, env):
		self.env = env
		categorical_q_net = categorical_q_network.CategoricalQNetwork(
			self.env.observation_spec()[0],
			env.action_spec()["action"],
			num_atoms=num_atoms,
			fc_layer_params=fc_layer_params)

		optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

		self.train_step_counter.assign(0)

		def splitter(obs):
			return obs

		self.agent = categorical_dqn_agent.CategoricalDqnAgent(
			self.env.time_step_spec(),
			self.env.action_spec()["action"],
			categorical_q_network=categorical_q_net,
			optimizer=optimizer,
			min_q_value=min_q_value,
			max_q_value=max_q_value,
			n_step_update=n_step_update,
			td_errors_loss_fn=common.element_wise_squared_loss,
			gamma=gamma,
			observation_and_action_constraint_splitter=splitter,
			train_step_counter=self.train_step_counter)

		self.agent.initialize()

		self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
			data_spec=self.agent.collect_data_spec,
			batch_size=train_env.batch_size,
			max_length=replay_buffer_capacity)

		self.dataset = self.replay_buffer.as_dataset(
			num_parallel_calls=3,
			sample_batch_size=batch_size,
			num_steps=n_step_update + 1).prefetch(3)

		self.iterator = iter(self.dataset)

		# print("self.agent.collect_data_spec", self.agent.collect_data_spec)

	def train(self):
		exp, _ = next(self.iterator)
		print(self.agent.training_data_spec)
		# with self._train_summary_writer.as_default():
		loss = self.agent.train(exp)

	def write_episode_rewards(self, step):
		with self._train_summary_writer.as_default():
			tf.summary.scalar(
				name="episode_rewards".format(self.player_index),
				data=self.episode_rewards,
				step=step)
			self._train_summary_writer.flush()

	def decide(self, game, playable_actions):
		"""Should return one of the playable_actions.

		Args:
			game (Game): complete game state. read-only.
			playable_actions (Iterable[Action]): options to choose from
		Return:
			action (Action): Chosen element of playable_actions
		"""
		# ===== YOUR CODE HERE =====
		# As an example we simply return the first action:

		# sample = create_sample(game, self.color)
		# vector = create_sample_vector(game, self.color)
		# print(len(sample.keys()), len(vector))
		# for key, val in zip(sample.keys(), vector):
		#     print(key, val)
		# mask = np.zeros(290)
		# player_actions = generate_playable_actions(game.state, self.color)
		# player_actions = list(map(to_action_space, player_actions))
		# mask[[player_actions]] = 1
		sample = create_sample(game, self.color)
		observation = np.array([sample[i] for i in feature_ordering], dtype=np.int64)
		mask = np.zeros(actions_count)
		mask[[to_action_space(action) for action in playable_actions]] = 1
		# print("Foo player", self.color, " is being made to pick an action from", np.sum(mask) // 1, "actions")

		step_type = StepType.MID if game.winning_color() is None else StepType.LAST
		reward = my_reward_function(game, self.color)


		time_step = tf_agents.trajectories.TimeStep(
			step_type=tf.expand_dims(step_type, axis=0),
			# step_type=tf.expand_dims(tf.cast(step_type, tf.int64), axis=0),
			reward=tf.expand_dims(tf.cast(reward, tf.float32), axis=0),
			discount=tf.expand_dims(tf.cast(discount, tf.float32), axis=0),
			observation=(tf.expand_dims(tf.cast(observation, tf.float64), axis=0), tf.expand_dims(tf.cast(mask, tf.float64), axis=0)))

		# print(time_step.step_type)
		# print(time_step.reward)
		# print(time_step.discount)
		# print(time_step.observation[0].shape)
		# print(time_step.observation[1].shape)
		# print("current_time_step", self.env.current_time_step())
		# print(self.env.time_step_spec())
		action_step = self.agent.collect_policy.action(time_step)
		# print("action_step_after")
		action_dict = {"action": action_step.action, "player_index": self.player_index}
		# print(action_step.action.numpy()[0])
		# print(from_action_space(action_step.action.numpy()[0], playable_actions))
		# now_time_step = self.env.current_time_step()
		def return_result(next_observation_tuple, next_reward, next_done, next_info):
			next_step_type = StepType.MID if next_done is None else StepType.LAST
			reward = my_reward_function(game, self.color)
			# print(next_observation)
			# print(next_reward)
			# print(next_done)
			# print(next_info)
			# next_time_step = self.env.current_time_step()
			# next_mask = np.zeros(actions_count)
			# next_mask[[to_action_space(action) for action in playable_actions]] = 1
			self.episode_rewards += reward
			next_observation, next_mask = next_observation_tuple

			batch_obs = (tf.expand_dims(tf.cast(next_observation, tf.float64), axis=0), tf.expand_dims(tf.cast(next_mask, tf.float64), axis=0))
			next_time_step = tf_agents.trajectories.TimeStep(
				# step_type=tf.expand_dims(tf.cast(step_type, tf.int64), axis=0),
				step_type=tf.expand_dims(next_step_type, axis=0),
				reward=tf.expand_dims(tf.cast(next_reward, tf.float32), axis=0),
				discount=tf.expand_dims(tf.cast(discount, tf.float32), axis=0),
				observation=batch_obs)
			traj = trajectory.from_transition(time_step, action_step, next_time_step)
			self.replay_buffer.add_batch(traj)

		return action_step.action.numpy()[0], return_result
		# return from_action_space(action_step.action.numpy()[0], playable_actions)
		# print("action_step.action", action_step.action)
		# print("action_dict_after")
		# next_time_step = self.env.step(action_dict)
		# traj = trajectory.from_transition(time_step, action_step, next_time_step)
		# print(action_step.action)
		# traj = trajectory.from_transition(time_step, action_step, next_time_step)
		# Add trajectory to the replay buffer
		# print(traj)
		# self.replay_buffer.add_batch(traj)
		# env.game.play_tick(env=env)
		# return playable_actions[0]
		# ===== END YOUR CODE =====

# def run_env(agent_list):
# 	py_env = suite_gym.load(
# 		environment_name=env_name,
# 		gym_kwargs={
# 		"config": {
# 			# "map_type": "MINI",
# 			# "vps_to_win": 6,
# 			"enemies": agent_list,
# 			"reward_function": my_reward_function,
# 			"representation": "vector"}})
# 	env = tf_py_environment.TFPyEnvironment(train_py_env)


agent_list = [FooPlayer(Color.RED, 0), FooPlayer(Color.WHITE, 1), FooPlayer(Color.BLUE, 2)]

gym_kwargs = {
	"config": {
		# "map_type": "MINI",
		# "vps_to_win": 6,
		"enemies": agent_list,
		"reward_function": my_reward_function,
		"representation": "vector",
	},
}

train_py_env = suite_gym.load(
	environment_name=env_name,
	gym_kwargs=gym_kwargs,
	discount=discount
)
# train_env = tf_py_environment.TFPyEnvironment(train_env)
eval_py_env = suite_gym.load(
	environment_name=env_name,
	gym_kwargs=gym_kwargs,
	discount=discount
)
# eval_env = tf_py_environment.TFPyEnvironment(eval_env)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
print('action_spec:', train_env.action_spec())
print('time_step_spec.observation:', train_env.time_step_spec().observation)
print('time_step_spec.step_type:', train_env.time_step_spec().step_type)
print('time_step_spec.discount:', train_env.time_step_spec().discount)
print('time_step_spec.reward:', train_env.time_step_spec().reward)

# print(train_env.current_time_step().observation[0].shape)
# print(train_env.current_time_step().observation[1].shape)
# print(train_env.current_time_step().reward.shape)
# print(train_env.current_time_step().discount.shape)
# print(train_env.current_time_step().step_type.shape)
new_obs = train_env.reset()
# print("new_obs", new_obs)
# print("step 1")
# train_env.step({"action": 0, "player_index": -1})
# print("step 2")
# train_env.step({"action": 0, "player_index": -1})

# for i in range(2000):
# 	train_env.step({"action": 0, "player_index": -1})
# 	if i % 50 == 0:
# 		# winner_key = player_key(game.state, color)
# 		# vp = game.state.player_state[f"{winner_key}_ACTUAL_VICTORY_POINTS"]
# 		# print(vp, color)
# 		pass
# [print(player.episode_rewards) for player in agent_list]

# q_net = CategoricalQNetwork(
#     input_tensor_spec=train_env.observation_spec(),
#     action_spec=train_env.action_spec(),
#     # self._observation_spec[self._player_index]["observation"],
#     # self._action_spec,
#     num_atoms=51,
#     fc_layer_params=(512, 512),
# )
# q_net.create_variables()


# class PTAgent:
# 	def __init__(self, player_index):
# 		self.player_index = player_index


def get_agent(player_index):
	categorical_q_net = categorical_q_network.CategoricalQNetwork(
		train_env.observation_spec()[0],
		train_env.action_spec()["action"],
		num_atoms=num_atoms,
		fc_layer_params=fc_layer_params)

	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

	train_step_counter = tf.Variable(0)

	def splitter(obs):
		# print("splitter:", player_index, np.sum(obs[1]))
		print("splitter:", player_index)
		return obs

	agent = categorical_dqn_agent.CategoricalDqnAgent(
		train_env.time_step_spec(),
		train_env.action_spec()["action"],
		categorical_q_network=categorical_q_net,
		optimizer=optimizer,
		min_q_value=min_q_value,
		max_q_value=max_q_value,
		n_step_update=n_step_update,
		td_errors_loss_fn=common.element_wise_squared_loss,
		gamma=gamma,
		observation_and_action_constraint_splitter=splitter,
		train_step_counter=train_step_counter)
	agent.initialize()

	replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
		data_spec=agent.collect_data_spec,
		batch_size=train_env.batch_size,
		max_length=replay_buffer_capacity)

	dataset = replay_buffer.as_dataset(
		num_parallel_calls=3, sample_batch_size=batch_size,
		num_steps=n_step_update + 1).prefetch(3)

	iterator = iter(dataset)

	return agent, replay_buffer, iterator


# agent_0, replay_buffer_0, iterator_0 = get_agent(0)
# agent_1, replay_buffer_1, iterator_1 = get_agent(1)
# agent_2, replay_buffer_2, iterator_2 = get_agent(2)


def compute_avg_return(environment, policy, player_index, num_episodes=10):

	total_return = 0.0
	for _ in range(num_episodes):

		time_step = environment._reset()
		episode_return = 0.0

		while not time_step.is_last():
			action_step = policy.action(time_step)
			# print("compute_avg_return, action_step", action_step)
			time_step = environment.step({"action": action_step.action, "player_index": player_index})
			episode_return += time_step.reward
		total_return += episode_return

	avg_return = total_return / num_episodes
	return avg_return.numpy()[0]

# print(train_env.time_step_spec())
# random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec()["action"])
#
# compute_avg_return(eval_env, random_policy, 0, num_eval_episodes)
# compute_avg_return(eval_env, random_policy, 1, num_eval_episodes)
# compute_avg_return(eval_env, random_policy, 2, num_eval_episodes)
# Please also see the metrics module for standard implementations of different
# metrics.
# replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
# 	data_spec=agent.collect_data_spec,
# 	batch_size=train_env.batch_size,
# 	max_length=replay_buffer_capacity)


def collect_step(environment, policy, player_index, replay_buffer):
	# print("test")
	time_step = environment.current_time_step()
	action_step = policy.action(time_step)
	action_dict = {"action": action_step.action, "player_index": player_index}
	# print("action_step.action", action_step.action)
	next_time_step = environment.step(action_dict)
	# traj = trajectory.from_transition(time_step, action_step, next_time_step)
	# print(action_step.action)
	traj = trajectory.from_transition(time_step, action_step, next_time_step)

	# Add trajectory to the replay buffer
	# print(traj)
	replay_buffer.add_batch(traj)


# for _ in range(initial_collect_steps):
# 	collect_step(train_env, random_policy)
# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.
# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1.
# dataset = replay_buffer.as_dataset(
# 	num_parallel_calls=3, sample_batch_size=batch_size,
# 	num_steps=n_step_update + 1).prefetch(3)
#
# iterator = iter(dataset)
# try:
#   %%time
# except:
#   pass
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
# agent_0.train = common.function(agent_0.train)
# agent_1.train = common.function(agent_1.train)
# agent_2.train = common.function(agent_2.train)
# Reset the train step
# agent_0.train_step_counter.assign(0)
# agent_1.train_step_counter.assign(0)
# agent_2.train_step_counter.assign(0)
# Evaluate the agent's policy once before training.
# avg_return_0 = compute_avg_return(eval_env, agent_0.policy, 0, num_eval_episodes)
# avg_return_1 = compute_avg_return(eval_env, agent_1.policy, 1, num_eval_episodes)
# avg_return_2 = compute_avg_return(eval_env, agent_2.policy, 2, num_eval_episodes)
# returns_0 = [avg_return_0]
# returns_1 = [avg_return_1]
# returns_2 = [avg_return_2]
# returns_0 = []
# returns_1 = []
# returns_2 = []
collect_steps = 0
episode_steps = 0

for _ in range(num_iterations):

	# Collect a few steps using collect_policy and save to the replay buffer.
	start = time.perf_counter()
	for i in range(collect_steps_per_iteration):
		collect_steps += 1
		episode_steps += 1
		ts = train_env.step({"action": 0, "player_index": -1})
		if i % 200 == 0:
			print([agent.episode_rewards for agent in agent_list])
		if ts.is_last():
			end = time.perf_counter()
			print("FINISHED", [agent.episode_rewards for agent in agent_list], "took", str(int(end - start)) + "s")
			for agent in agent_list:
				agent.write_episode_rewards(collect_steps)
				agent.episode_rewards = 0
			player_rewards = {
				Color.RED: 0,
				Color.WHITE: 0,
				Color.BLUE: 0
			}
			start = time.perf_counter()

	for _ in range(train_steps_per_iteration):
		for agent in agent_list:
			agent.train()

		# if step_0 % log_interval == 0:
		# 	print('step = {0}: loss = {1}'.format(step_0, train_loss_0.loss))
		#
		# if step_0 % eval_interval == 0:
		# 	avg_return = compute_avg_return(eval_env, agent_0.policy, num_eval_episodes)
		# 	print('step = {0}: Average Return = {1:.2f}'.format(step_0, avg_return))
		# 	returns_0.append(avg_return)

# # steps = range(0, num_iterations + 1, eval_interval)
# # plt.plot(steps, returns)
# plt.ylabel('Average Return')
# plt.xlabel('Step')
# plt.ylim(top=550)
# plt.show()

train_env.close()
eval_env.close()
