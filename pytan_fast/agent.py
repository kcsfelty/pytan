import os
import time

import tensorflow as tf
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.policies import PolicySaver
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
				 learning_rate=0.005,
				 batch_size=12000,
				 replay_buffer_capacity=12000,
				 num_atoms=51,
				 fc_layer_params=(2**6, 2**6),
				 min_q_value=-1,
				 max_q_value=1,
				 n_step_update=18,
				 gamma=0.92,
				 epsilon_greedy=0.05,
				 file_dir="./logs"
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
		self.file_dir = file_dir
		self.observers = [self.write_summary]

		self.train_step_counter = tf.Variable(0, dtype=tf.int64)

		self.categorical_q_net = categorical_q_network.CategoricalQNetwork(
			env_specs["env_observation_spec"],
			action_spec=env_specs["env_action_spec"],
			num_atoms=self.num_atoms,
			fc_layer_params=self.fc_layer_params,
			name="agent{}_network".format(player_index)
		)

		self.agent = categorical_dqn_agent.CategoricalDqnAgent(
			time_step_spec=env_specs["env_time_step_spec"],
			action_spec=env_specs["env_action_spec"],
			categorical_q_network=self.categorical_q_net,
			optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
			min_q_value=self.min_q_value,
			max_q_value=self.max_q_value,
			n_step_update=self.n_step_update,
			td_errors_loss_fn=common.element_wise_huber_loss,
			gamma=self.gamma,
			epsilon_greedy=self.epsilon_greedy,
			observation_and_action_constraint_splitter=splitter,
			train_step_counter=self.train_step_counter,
			summarize_grads_and_vars=True
		)

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

		# self._train_checkpointer = common.Checkpointer(
		# 	ckpt_dir="./training/agent" + str(player_index) + "/train",
		# 	agent=self.agent,
		# 	global_step=self.write_count,
		# 	metrics=metric_utils.MetricsGroup(self.train_metrics, 'train_metrics'),
		# 	max_to_keep=1)

		# self._policy_checkpointer = common.Checkpointer(
		# 	ckpt_dir=os.path.join("./training/agent" + str(player_index), 'policy'),
		# 	policy=self.agent.policy,
		# 	max_to_keep=1)

		self.saver = PolicySaver(self.agent.policy)
		self.log_dir = log_dir + "/agent" + str(player_index)
		self.writer = tf.compat.v2.summary.create_file_writer(self.log_dir)

	def write_summary(self, summaries, step):
		with self.writer.as_default():
			for summary_key in summaries["scalars"]:
				tf.summary.scalar(name=summary_key, data=summaries["scalars"][summary_key], step=step.item())
			for summary_key in summaries["histograms"]:
				tf.summary.histogram(name=summary_key, data=summaries["histograms"][summary_key], step=step.item(), buckets=len(summaries["histograms"][summary_key]))

	def train(self, iterations):
		print("Agent{} losses:".format(str(self.player_index)))
		for j in range(iterations):
			exp, _ = next(self.iterator)
			with self.writer.as_default():
				self.agent.train(exp)
		self.saver.save(self.log_dir)

