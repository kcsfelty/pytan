import asyncio
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
		train_process_count=8,
		eval_process_count=2,
		thread_count=2 ** 5,
		agent_count=1,

		# Batching
		train_game_count=2 ** 7,
		eval_game_count=2 ** 3,
		n_step_update=32,

		# Replay buffer
		replay_buffer_size=2 * goal_player_steps + 1,
		replay_batch_size=2 ** 3,

		# Network parameters
		learn_rate=1e-4,
		fc_layer_params=(2 ** 10, 2 ** 9, 2 ** 8, 2 ** 7,),
		gamma=n_step_gamma,

		# Greedy policy epsilon
		epsilon_greedy_start=0.05,
		epsilon_greedy_end=0.01,
		epsilon_greedy_half_life=10e6,

		# Intervals
		total_steps=500e6,
		eval_steps=2 ** 4 * 10000,
		train_steps=2 ** 3,
		train_per_eval=2 ** 11,
		train_log_interval=2 ** 7,
		eval_log_interval=2 ** 8,
	):
	train_iteration = tf.Variable(0, dtype=tf.int32)
	eval_iteration = tf.Variable(0, dtype=tf.int32)
	train_global_step = tf.Variable(0, dtype=tf.int32)
	eval_global_step = tf.Variable(0, dtype=tf.int32)
	epsilon_greedy_delta = tf.constant(epsilon_greedy_start - epsilon_greedy_end)
	epsilon_greedy_base = tf.constant(1 - math.log(2) / epsilon_greedy_half_life)
	train_writer = tf.summary.create_file_writer(logdir=os.path.join(log_dir, "game"))
	eval_writer = tf.summary.create_file_writer(logdir=os.path.join(log_dir, "eval"))

	def epsilon_greedy():
		epsilon = tf.math.pow(epsilon_greedy_base, tf.cast(train_global_step, tf.float32))
		epsilon = tf.multiply(epsilon, epsilon_greedy_delta)
		epsilon = tf.add(epsilon_greedy_end, epsilon)
		return epsilon

	def get_env(process_count, game_count, env_id_prefix):
		meta_env_list = []
		for i in range(process_count):
			class ParallelPyTan(PyTan, ABC):
				def __init__(self):
					super().__init__(
						game_count=game_count,
						worker_count=thread_count,
						log_dir=log_dir,
						env_id="{}-{}".format(env_id_prefix, i))

			meta_env_list.append(ParallelPyTan)
		return tf_py_environment.TFPyEnvironment(ParallelPyEnvironment(meta_env_list))

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
			epsilon_greedy=epsilon_greedy,
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
		existing_games = len(tf.where(~traj_list[0].is_first()))
		eval_global_step.assign_add(existing_games)
		eval_iteration.assign_add(1)

	def train_policies():
		for agent in agent_list:
			agent.train()

	def eval_log():
		if eval_iteration.numpy() % eval_log_interval == 0:
			log_str = ""
			log_str += "[EVAL]  "
			log_str += "[global: {}] ".format(str(eval_global_step.numpy()).rjust(10))
			log_str += "[pct: {}%] ".format(str(int((eval_global_step.numpy() % eval_steps) / eval_steps * 100)).rjust(5))
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
			print(log_str)

	metrics = ObservationMetrics()

	train_env = get_env(
		process_count=train_process_count,
		game_count=train_game_count,
		env_id_prefix="train")

	eval_env = get_env(
		process_count=eval_process_count,
		game_count=eval_game_count,
		env_id_prefix="eval")

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
		policy_list=[agent.agent.collect_policy for agent in agent_list],
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
		policy_list=[agent.agent.policy for agent in agent_list],
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
	while train_iteration.numpy() < replay_buffer_size:
		train_driver.run()

	print("Initial steps completed. Beginning Training.")
	while train_global_step.numpy() < total_steps:
		for _ in range(train_per_eval):
			train_driver.run()
			train_policies()
		print("Performing evaluation.")
		eval_env.reset()
		eval_driver.run()


def main(_):
	mapped_train_eval()


if __name__ == '__main__':
	tf_agents.system.multiprocessing.handle_main(main)
