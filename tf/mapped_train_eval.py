import math
import os.path
from abc import ABC

import tf_agents
from tf_agents.environments import tf_py_environment, ParallelPyEnvironment
from tf_agents.train.step_per_second_tracker import StepPerSecondTracker

from game.game import PyTan
from reference.settings import player_count
from tf.observation_metrics import ObservationMetrics
from tf.agent import Agent
from tf.mapped_driver import MappedDriver
from tf.random_policy_mapping import random_policy_mapping
import tensorflow as tf
import numpy as np

goal_episode_steps = 1200
goal_player_steps = goal_episode_steps // player_count
oldest_n_step_discount_factor = 3
half_life_steps = goal_player_steps / oldest_n_step_discount_factor
n_step_gamma = 1 - math.log(2) / half_life_steps

num_game = 2 ** 12
num_process = 6
game_per_process = num_game // num_process


def mapped_train_eval(
		# Performance / Logging
		log_dir=os.path.join("./logs", "current"),
		train_process_count=7,
		eval_process_count=4,
		thread_count=2 ** 5,
		agent_count=1,

		# Train config
		fill_replay_buffer=True,
		initial_intervals=2000,

		# Batching
		train_game_count=2 ** 8,
		eval_game_count=2 ** 2,
		n_step_update=2 ** 6,

		# Replay buffer
		replay_buffer_size=500,
		replay_batch_size=2 ** 6,

		# Network parameters
		learn_rate=1e-3,
		fc_layer_params=(2 ** 8, 2 ** 8, 2 ** 8, 2 ** 7),
		gamma=n_step_gamma,

		# Greedy policy epsilon
		scalar_epsilon=None,
		epsilon_start=0.00,
		epsilon_end=0.00,
		epsilon_half_life=1,

		# Intervals
		total_steps=1e9,
		eval_steps=1e6,
		eval_episodes=64,
		train_steps=2 ** 6,
		train_per_eval=2 ** 11,
		train_log_interval=2 ** 8,
		eval_log_interval=2 ** 8,
	):
	train_iteration = tf.Variable(0, dtype=tf.int32)
	eval_iteration = tf.Variable(0, dtype=tf.int32)
	train_global_step = tf.Variable(0, dtype=tf.int32)
	eval_global_step = tf.Variable(0, dtype=tf.int32)
	current_eval_episodes = tf.Variable(0, dtype=tf.int32)
	train_writer = tf.summary.create_file_writer(logdir=os.path.join(log_dir, "game"))
	eval_writer = tf.summary.create_file_writer(logdir=os.path.join(log_dir, "eval"))

	epsilon_delta = tf.constant(epsilon_start - epsilon_end)
	epsilon_base = tf.constant(1 - math.log(2) / epsilon_half_life)

	eval_turn_list = []

	def step_epsilon():
		eps = tf.math.pow(epsilon_base, tf.cast(train_global_step, tf.float32))
		eps = tf.multiply(eps, epsilon_delta)
		eps = tf.add(epsilon_end, eps)
		return eps

	def get_env(process_count, game_count):
		class ParallelPyTan(PyTan, ABC):
			def __init__(self):
				super().__init__(game_count=game_count, worker_count=thread_count, log_dir=log_dir)

		meta_env_list = [ParallelPyTan] * process_count
		return tf_py_environment.TFPyEnvironment(ParallelPyEnvironment(meta_env_list))

	# TODO modify to allow for more than one type of agent, also build a base class BasePyTanAgent, C51PyTanAgent, etc.
	def get_agent_list():
		return [Agent(
			index=index,
			action_spec=train_env.action_spec(),
			time_step_spec=train_env.time_step_spec(),
			batch_size=train_policy_mapping.shape[1],
			replay_buffer_size=replay_buffer_size,
			replay_batch_size=replay_batch_size,
			fc_layer_params=fc_layer_params,
			n_step_update=n_step_update,
			learn_rate=learn_rate,
			epsilon_greedy=scalar_epsilon or step_epsilon,
			gamma=gamma,
			log_dir=os.path.join(log_dir, "agents")
		) for index in range(agent_count)]

	def summarize_with_writer(writer):
		def handle_summaries(traj_list):
			for agent, traj in zip(agent_list, traj_list):
				finished_games = tf.where(traj.is_last())
				observation, _ = traj.observation
				observation = tf.squeeze(tf.gather(observation, finished_games), axis=(1,))
				if observation.shape[0]:
					game_metrics, agent_metrics = metrics.summarize(observation)
					with tf.name_scope("summary"):
						with writer.as_default(step=train_global_step.numpy()):
							for game_metric in game_metrics:
								data = game_metrics[game_metric].numpy()
								for scalar in data:
									tf.summary.scalar(name=game_metric.lower(), data=scalar)

						with agent.writer.as_default(step=train_global_step.numpy()):
							for agent_metric in agent_metrics:
								data = agent_metrics[agent_metric]
								for scalar in data:
									tf.summary.scalar(name=agent_metric.lower(), data=scalar)
		return handle_summaries

	def add_batch(traj_list):
		for agent, traj in zip(agent_list, traj_list):
			agent.replay_buffer.add_batch(traj)

	def update_train_global_step(traj_list):
		existing_games = len(tf.where(~traj_list[0].is_first()))
		train_global_step.assign_add(existing_games)
		train_iteration.assign_add(1)

	def update_eval_global_step(traj_list):
		games_completed = np.sum(traj_list[0].is_last())
		current_eval_episodes.assign_add(games_completed)
		existing_games = len(tf.where(~traj_list[0].is_first()))
		eval_global_step.assign_add(existing_games)
		eval_iteration.assign_add(1)

	def process_eval_results():
		# TODO: take a statistically significant number of samples (ideally, more) from the eval driver
		# then calculate the mean and std
		# compare the results to the found random mean / std and estimated human mean / std
		pass

	def train_policies():

		# TODO: train async to minimize performance hit

		for agent in agent_list:
			agent.train()

	def eval_log():
		if eval_iteration.numpy() % eval_log_interval == 0:
			log_str = ""
			log_str += "[EVAL]  "
			log_str += "[global: {}] ".format(str(eval_global_step.numpy()).rjust(10))
			log_str += "[games: {}%] ".format(str(int(current_eval_episodes.numpy() / eval_episodes * 100)).rjust(5))
			log_str += "[rate: {}] ".format(str(eval_rate_tracker.steps_per_second())[:7])
			print(log_str)

	def train_log():
		if train_iteration.numpy() % train_log_interval == 0:
			log_str = ""
			log_str += "[TRAIN]  "
			log_str += "[global: {}] ".format(str(train_global_step.numpy()).rjust(10))
			log_str += "[pct: {}%] ".format(str(int(train_global_step.numpy() / total_steps * 100)).rjust(5))
			log_str += "[step rate: {}] ".format(str(train_rate_tracker.steps_per_second())[:7])
			log_str += "[iter rate: {}] ".format(str(iteration_rate_tracker.steps_per_second())[:7])
			log_str += "[epsilon: {}] ".format(str(step_epsilon().numpy()))
			print(log_str)

	metrics = ObservationMetrics()

	train_env = get_env(
		process_count=train_process_count,
		game_count=train_game_count)

	eval_env = get_env(
		process_count=eval_process_count,
		game_count=eval_game_count)

	train_policy_mapping = random_policy_mapping(
		agent_count=agent_count,
		player_count=player_count,
		game_count=train_game_count,
		process_count=train_process_count)

	eval_policy_mapping = random_policy_mapping(
		agent_count=agent_count,
		player_count=player_count,
		game_count=eval_game_count,
		process_count=eval_process_count)

	agent_list = get_agent_list()

	iteration_rate_tracker = StepPerSecondTracker(train_iteration)
	train_rate_tracker = StepPerSecondTracker(train_global_step)
	eval_rate_tracker = StepPerSecondTracker(eval_global_step)

	train_driver = MappedDriver(
		environment=train_env,
		policy_list=[agent.collect_policy for agent in agent_list],
		policy_mapping=train_policy_mapping,
		max_steps=train_steps,
		observers=[
			update_train_global_step,
			add_batch,
			summarize_with_writer(train_writer),
			lambda _: train_log(),
		])

	eval_driver = MappedDriver(
		environment=eval_env,
		policy_list=[agent.policy for agent in agent_list],
		policy_mapping=eval_policy_mapping,
		max_steps=eval_steps,
		observers=[
			update_eval_global_step,
			summarize_with_writer(eval_writer),
			lambda _: eval_log(),
		])

	print("Performing initial evaluation.")
	eval_driver.run()

	print("Providing initial steps to seed replay buffer.")
	while train_iteration.numpy() < replay_buffer_size or train_iteration.numpy() < initial_intervals:
		train_driver.run()

	print("Initial steps completed. Beginning Training.")
	while train_global_step.numpy() < total_steps:
		for _ in range(train_per_eval):
			train_driver.run()
			train_policies()
		print("Performing evaluation.")
		current_eval_episodes.assign(0)
		eval_env.reset()
		eval_driver.run()


def main(_):
	mapped_train_eval()


if __name__ == '__main__':
	tf_agents.system.multiprocessing.handle_main(main)
