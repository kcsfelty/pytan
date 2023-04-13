from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.distributions import reparameterized_sampling
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory, PolicyStep
from tf_agents.utils import common
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from pytan_fast.specs import obs_type_spec, fake_action_spec, fake_time_step_spec

support = np.linspace(-1, 1, 51, dtype=np.float32)
neg_inf = tf.constant(-np.inf, dtype=tf.float32)
seed_stream = tfp.util.SeedStream(seed=None, salt='tf_agents_tf_policy')


class Agent:
	def __init__(self,
				 q_min=-1,
				 q_max=1,
				 n_step_update=1,
				 replay_buffer_size=1000,
				 learn_rate=0.0001,
				 fc_layer_params=(2 ** 6, 2 ** 6, 2 ** 6,),
				 game_count=1,
				 replay_batch_size=250,
				 ):
		self.categorical_q_net = categorical_q_network.CategoricalQNetwork(
			input_tensor_spec=obs_type_spec,
			action_spec=fake_action_spec,
			fc_layer_params=fc_layer_params)

		self.train_counter = tf.Variable(0, dtype=tf.int32)

		self.agent = categorical_dqn_agent.CategoricalDqnAgent(
			time_step_spec=fake_time_step_spec,
			action_spec=fake_action_spec,
			categorical_q_network=self.categorical_q_net,
			train_step_counter=self.train_counter,
			optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learn_rate),
			n_step_update=n_step_update,
			min_q_value=q_min,
			max_q_value=q_max,
			summarize_grads_and_vars=True,
			observation_and_action_constraint_splitter=self.splitter)

		self.replay_buffer = TFUniformReplayBuffer(
			data_spec=self.agent.collect_data_spec,
			batch_size=game_count,
			max_length=replay_buffer_size
		)

		self.dataset = self.replay_buffer.as_dataset(
			num_parallel_calls=3,
			sample_batch_size=replay_batch_size,
			num_steps=n_step_update + 1 if n_step_update else 1,
		).prefetch(3)

		self.iterator = iter(self.dataset)

		self.game_count = game_count
		self.time_step = None
		self.action = None

	@tf.function
	def act(self, time_step):
		obs, mask = time_step.observation
		activations = self.categorical_q_net(obs)[0]
		q_values = common.convert_q_logits_to_values(activations, support)
		logits = tf.compat.v2.where(tf.cast(mask, tf.bool), q_values, neg_inf)
		dist = tfp.distributions.Categorical(logits=logits, dtype=tf.float32)
		action = tf.nest.map_structure(self.sample_logit_distribution, dist)
		action = tf.cast(action, dtype=tf.int32)
		action = PolicyStep(action)
		self.action = action
		return action

	def add_batch(self, next_time_step):
		if not self.time_step:
			self.time_step = next_time_step
			return
		self.replay_buffer.add_batch(trajectory.from_transition(
			self.time_step,
			self.action,
			next_time_step))
		self.time_step = next_time_step

	def train(self):
		if self.replay_buffer.num_frames() > self.game_count:
			exp, _ = next(self.iterator)
			self.agent.train(exp)

	@staticmethod
	def sample_logit_distribution(distribution):
		return reparameterized_sampling.sample(distribution, seed=seed_stream())

	@staticmethod
	def splitter(obs_tuple):
		obs, mask = obs_tuple
		return obs, mask
