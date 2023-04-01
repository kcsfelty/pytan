import math
import os
import time

import tensorflow as tf
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.policies import PolicySaver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


def splitter(obs_tuple):
	obs, mask = obs_tuple
	return obs, mask


class FastAgent:
	def __init__(self,
				 env_specs,
				 player_index,
				 log_dir,
				 global_step,
				 train_interval=1,
				 learning_rate=0.001,
				 batch_size=1,
				 replay_buffer_capacity=1000,
				 num_atoms=51 * 1,
				 fc_layer_params=(2**5, 2**5),
				 min_q_value=0,
				 max_q_value=10,
				 n_step_update=1,
				 gamma=0.95,
				 epsilon_greedy=None,
				 eps_min=0.15,
				 eps_start=0.85,
				 eps_decay_rate=0.9999,
				 checkpoint_dir="checkpoints",
				 checkpoint_interval=10000,
				 eval_interval=10000,
				 min_train_frames=1000,
		 ):
		self.player_index = player_index
		self.batch_size = batch_size
		self.replay_buffer_capacity = replay_buffer_capacity
		self.num_atoms = num_atoms
		self.fc_layer_params = fc_layer_params
		self.min_q_value = min_q_value
		self.max_q_value = max_q_value
		self.n_step_update = n_step_update
		self.gamma = gamma
		self.epsilon_greedy = epsilon_greedy
		self.agent_prefix = "agent{}".format(str(player_index))
		self.train_interval = train_interval
		self.checkpoint_interval = checkpoint_interval
		self.global_step = global_step
		self.eval_interval = eval_interval
		self.eps_min = eps_min
		self.eps_start = eps_start
		self.eps_decay_rate = eps_decay_rate
		self.eps_cof = 1.
		self.epsilon = self.eps_start + self.eps_min
		self.min_train_frames = min_train_frames
		self.losses = []

		self.train_step_counter = tf.Variable(0, dtype=tf.int64)
		self.step_counter = tf.Variable(0, dtype=tf.int64)

		self.categorical_q_net = categorical_q_network.CategoricalQNetwork(
			env_specs["env_observation_spec"],
			action_spec=env_specs["env_action_spec"],
			num_atoms=self.num_atoms,
			fc_layer_params=self.fc_layer_params,
			name=self.agent_prefix + "_network")

		# self.target_categorical_q_net = categorical_q_network.CategoricalQNetwork(
		# 	env_specs["env_observation_spec"],
		# 	action_spec=env_specs["env_action_spec"],
		# 	num_atoms=self.num_atoms,
		# 	fc_layer_params=self.fc_layer_params,
		# 	name=self.agent_prefix + "_target_network")

		self.agent = categorical_dqn_agent.CategoricalDqnAgent(
			time_step_spec=env_specs["env_time_step_spec"],
			action_spec=env_specs["env_action_spec"],
			categorical_q_network=self.categorical_q_net,
			# target_categorical_q_network=self.target_categorical_q_net,
			optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
			min_q_value=self.min_q_value,
			max_q_value=self.max_q_value,
			n_step_update=self.n_step_update,
			td_errors_loss_fn=common.element_wise_huber_loss,
			gamma=self.gamma,
			epsilon_greedy=self.epsilon_greedy or self.get_epsilon,
			observation_and_action_constraint_splitter=splitter,
			train_step_counter=self.train_step_counter,
			summarize_grads_and_vars=True,
			debug_summaries=True,)

		# self.categorical_q_net.summary()
		self.agent.initialize()

		self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
			data_spec=self.agent.collect_data_spec,
			batch_size=1,
			max_length=self.replay_buffer_capacity)

		self.dataset = self.replay_buffer.as_dataset(
			num_parallel_calls=3,
			sample_batch_size=self.batch_size,
			num_steps=self.n_step_update + 1).prefetch(3)

		self.iterator = iter(self.dataset)
		self.observers = [self.add_step]

		self.checkpointer = common.Checkpointer(
			ckpt_dir=os.path.join(checkpoint_dir, self.agent_prefix),
			max_to_keep=1,
			policy=self.agent.policy,
			replay_buffer=self.replay_buffer,
			train_step_counter=self.train_step_counter,
			step_counter=self.step_counter)

		self.checkpointer.initialize_or_restore()

		self.log_dir = log_dir
		self.writer = tf.compat.v2.summary.create_file_writer(os.path.join(self.log_dir, self.agent_prefix))

	def get_epsilon(self):
		return self.epsilon

	def update_epsilon(self):
		if self.replay_buffer.num_frames() < self.min_train_frames:
			self.epsilon = 1.
		self.eps_cof *= self.eps_decay_rate
		self.epsilon = self.eps_min + self.eps_cof * self.eps_start

	def add_step(self, step):
		self.replay_buffer.add_batch(step)
		self.step_counter.assign_add(1)
		self.update_epsilon()

		if not self.replay_buffer.num_frames() < self.min_train_frames:
			if self.step_counter.numpy() % self.train_interval == 0:
				self.train()

		if self.step_counter.numpy() % self.checkpoint_interval == 0:
			print("Checkpointing", self.agent_prefix, "current step:", self.step_counter.read_value(),"current eps:", self.epsilon)
			self.checkpoint()

		if self.step_counter.numpy() % self.eval_interval == 0:
			print("Evaluating", self.agent_prefix)
			self.eval()

	def act(self, time_step, collect=True):
		if collect:
			return self.agent.collect_policy.action(time_step)
		return self.agent.policy.action(time_step)

	def write_summary(self, summaries, step):
		with self.writer.as_default():
			for summary_key in summaries["scalars"]:
				tf.summary.scalar(
					name=summary_key,
					data=summaries["scalars"][summary_key],
					step=step.numpy().item())
			for summary_key in summaries["histograms"]:
				tf.summary.histogram(
					name=summary_key,
					data=summaries["histograms"][summary_key],
					step=step.numpy().item(),
					buckets=len(summaries["histograms"][summary_key]))

	def train(self):
		exp, _ = next(self.iterator)
		with self.writer.as_default():
			loss_info = self.agent.train(exp)
			self.losses.append(loss_info.loss)
			if self.check_loss_divergence():
				self.reduce_n()

	def check_loss_divergence(self):
		lookback = 1000
		count = 3
		tol = 0.01
		if len(self.losses) < lookback + count:
			return False
		for i in range(count):
			if not math.fabs(self.losses[-lookback] - self.losses[-(i + 1)]) < tol:
				return False
		return True

	def reduce_n(self, amount=1):
		min_n = 2
		if self.n_step_update > min_n:
			self.n_step_update -= amount
			print(self.agent_prefix, "reducing n to", self.n_step_update, "current step", self.step_counter)
			self.dataset = self.replay_buffer.as_dataset(
				num_parallel_calls=3,
				sample_batch_size=self.batch_size,
				num_steps=self.n_step_update + 1).prefetch(3)

			self.iterator = iter(self.dataset)
			self.agent._train_sequence_length -= 1

	def checkpoint(self):
		self.checkpointer.save(global_step=self.global_step.numpy().item())

	def eval(self):
		pass
