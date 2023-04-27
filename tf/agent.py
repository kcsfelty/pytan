from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import tensorflow as tf


class Agent:
	def __init__(self,
			index,
			time_step_spec,
			action_spec,
			q_min=-1,
			q_max=1,
			n_step_update=1,
			replay_buffer_size=1000,
			learn_rate=0.0001,
			fc_layer_params=(2 ** 6, 2 ** 6, 2 ** 6,),
			epsilon_greedy=0.1,
			gamma=1.,
			batch_size=1,
			replay_batch_size=250,
			log_dir="./logs"
		):

		obs_spec, _ = time_step_spec.observation

		self.categorical_q_net = categorical_q_network.CategoricalQNetwork(
			input_tensor_spec=obs_spec,
			action_spec=action_spec,
			fc_layer_params=fc_layer_params)

		self.train_counter = tf.Variable(0, dtype=tf.int64)

		self.agent = categorical_dqn_agent.CategoricalDqnAgent(
			time_step_spec=time_step_spec,
			action_spec=action_spec,
			categorical_q_network=self.categorical_q_net,
			optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learn_rate),
			observation_and_action_constraint_splitter=self.splitter,
			min_q_value=q_min,
			max_q_value=q_max,
			epsilon_greedy=epsilon_greedy,
			n_step_update=n_step_update,
			train_step_counter=self.train_counter,
			target_update_tau=1.0,
			target_update_period=1,
			gamma=gamma,
			summarize_grads_and_vars=True)

		self.train_fn = common.function(self.agent.train)

		self.replay_buffer = TFUniformReplayBuffer(
			data_spec=self.agent.collect_data_spec,
			batch_size=batch_size,
			max_length=replay_buffer_size
		)

		self.replay_batch_size = replay_batch_size
		self.dataset = self.replay_buffer.as_dataset(
			num_parallel_calls=3,
			sample_batch_size=self.replay_batch_size,
			num_steps=n_step_update + 1 if n_step_update else 1,
		).prefetch(3)

		self.iterator = iter(self.dataset)

		self.writer = tf.summary.create_file_writer(logdir=log_dir + "/agent{}".format(index))
		self.time_step = None
		self.action = None

	def train(self):
		exp, _ = next(self.iterator)
		with self.writer.as_default():
			self.train_fn(exp)

	@staticmethod
	def splitter(obs_tuple):
		obs, mask = obs_tuple
		return obs, mask
