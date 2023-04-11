import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import TensorSpec
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.distributions import reparameterized_sampling
from tf_agents.environments import tf_py_environment
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories import TimeStep, trajectory, PolicyStep
from tf_agents.utils import common

from pytan_fast.game import PyTanFast
from pytan_fast.settings import player_count

support = np.linspace(-1, 1, 51, dtype=np.float32)
neg_inf = tf.constant(-np.inf, dtype=tf.float32)
seed_stream = tfp.util.SeedStream(seed=None, salt='tf_agents_tf_policy')


class MetaAgent:
	def __init__(self, agent_list, game_count):
		self.agent_list = agent_list
		self.agent_count = len(self.agent_list)
		self.game_count = game_count

	def train(self):
		for agent in self.agent_list:
			agent.train()

	def act(self, time_step_list):
		action_list = []
		time_step_list = [TimeStep(*time_step_list[i:i+4]) for i in range(0, len(time_step_list), 4)]
		for agent, time_step in zip(self.agent_list, time_step_list):
			action_list.append(agent.act(time_step))
		# actions = tf.map_fn(
		# 	lambda a, ts: a.act(ts),
		# 	(self.agent_list, time_step),
		# 	fn_output_signature=tf.TensorSpec((self.agent_count, self.game_count,), dtype=tf.float32))
		# actions = tf.cast(actions, dtype=tf.int32)
		return action_list

	# @tf.function
	# def act_agent(self, agent, time_step):
	# 	return agent.act(time_step)
	#
	# @tf.function
	# def split_time_step(self, time_step):
	# 	obs, mask = time_step.observation
	# 	split_observation = tf.split(obs, self.agent_count)
	# 	split_mask = tf.split(mask, self.agent_count)
	# 	split_step_type = tf.split(time_step.step_type, self.agent_count)
	# 	split_discount = tf.split(time_step.discount, self.agent_count)
	# 	split_reward = tf.split(time_step.reward, self.agent_count)
	# 	return tf.map_fn(
	# 		self.merge_time_step,
	# 		split_observation,
	# 		split_mask,
	# 		split_step_type,
	# 		split_discount,
	# 		split_reward,
	# 		fn_output_signature=tf.TensorSpec((self.game_count,), dtype=tf.float32))
	#
	# @tf.function
	# def merge_time_step(self, split_observation, split_mask, split_step_type, split_discount, split_reward):
	# 	return TimeStep(
	# 		step_type=split_step_type,
	# 		reward=split_reward,
	# 		discount=split_discount,
	# 		observation=(split_observation, split_mask)
	# 	)


class Agent:
	def __init__(self,
				 q_min=-1,
				 q_max=1,
				 n_step_update=1,
				 replay_buffer_size=1000,
				 learn_rate=0.00001,
				 fc_layer_params=(2 ** 8, 2 ** 8, 2 ** 7, 2 ** 7, 2 ** 6, 2 ** 6,),
				 game_count=1,
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
			sample_batch_size=game_count,
			num_steps=n_step_update + 1,
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


action_count = 379
observation_count = 1402
fake_action_spec = BoundedTensorSpec(
	shape=(),
	dtype=tf.int32,
	name='action_mask',
	minimum=np.array(0),
	maximum=np.array(action_count - 1))
step_type_spec = TensorSpec(
	shape=(),
	dtype=tf.int32,
	name='step_type')
reward_type_spec = BoundedTensorSpec(
	shape=(),
	dtype=tf.float32,
	name='reward',
	minimum=np.array(-1., dtype=np.float32),
	maximum=np.array(1., dtype=np.float32))
discount_type_spec = BoundedTensorSpec(
	shape=(),
	dtype=tf.float32,
	name='discount',
	minimum=np.array(0., dtype=np.float32),
	maximum=np.array(1., dtype=np.float32))
obs_type_spec = BoundedTensorSpec(
	shape=(observation_count,),
	dtype=tf.int32,
	name='observation',
	minimum=np.array(0),
	maximum=np.array(128))
mask_type_spec = BoundedTensorSpec(
	shape=(action_count,),
	dtype=tf.int32,
	name='action_mask',
	minimum=np.array(0),
	maximum=np.array(1))
fake_time_step_spec = TimeStep(
	step_type_spec,
	reward_type_spec,
	discount_type_spec,
	(obs_type_spec, mask_type_spec))


def train_eval(
		game_count=1000,
		total_steps=1e9,
		train_interval=1,
		eval_interval=1,
		log_interval=1e5,
	):

	def maybe_train():
		if global_step.numpy() % train_interval == 0:
			meta.train()

	def maybe_eval():
		if global_step.numpy() % eval_interval == 0:
			pass

	def maybe_log():
		if global_step.numpy() % log_interval == 0:
			step = global_step.numpy().item()
			log_str = ""
			log_str += "[global: {}]".format(str(step).rjust(10))
			log_str += "\t"
			log_str += "[pct: {}%]".format(str(int(step / total_steps * 100)))
			print(log_str)

	def run():
		time_step = env.current_time_step()
		while global_step.numpy() < total_steps:
			action = meta.act(time_step)
			time_step = env.step(action)
			maybe_train()
			maybe_eval()
			maybe_log()

	global_step = tf.Variable(0, dtype=tf.int32)
	game = PyTanFast(game_count=game_count, global_step=global_step)
	env = tf_py_environment.TFPyEnvironment(game)
	agent_list = [Agent(game_count=game_count) for _ in range(player_count)]
	meta = MetaAgent(agent_list, game_count)
	run()


if __name__ == "__main__":
	train_eval()
