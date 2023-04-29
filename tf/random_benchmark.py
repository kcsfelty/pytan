import os.path
from abc import ABC

import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.environments import tf_py_environment, ParallelPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.train.step_per_second_tracker import StepPerSecondTracker

from game.game import PyTan
from reference.settings import player_count
from tf.mapped_driver import MappedDriver
from tf.observation_metrics import ObservationMetrics
from tf.random_policy_mapping import random_policy_mapping
from reference.definitions import turn_number


def random_benchmark(
		# Performance / Logging
		log_dir=os.path.join("./logs", "random"),
		train_process_count=7,
		thread_count=2 ** 5,
		agent_count=1,

		# Batching
		train_game_count=2 ** 8,

		# Intervals
		total_episodes=10000,
	):
	iteration = tf.Variable(0, dtype=tf.int32)
	train_global_step = tf.Variable(0, dtype=tf.int32)
	episodes = tf.Variable(0, dtype=tf.int32)
	episode_turn_list = []

	summary_writer = tf.summary.create_file_writer(logdir=os.path.join(log_dir, "game"))

	def get_env(process_count, game_count):
		class ParallelPyTan(PyTan, ABC):
			def __init__(self):
				super().__init__(game_count=game_count, worker_count=thread_count, log_dir=log_dir)
		meta_env_list = [ParallelPyTan] * process_count
		return tf_py_environment.TFPyEnvironment(ParallelPyEnvironment(meta_env_list))

	def update_train_global_step(traj_list):
		existing_games = len(tf.where(~traj_list[0].is_first()))
		train_global_step.assign_add(existing_games)
		iteration.assign_add(1)

	def train_log(traj_list):
		games_completed = np.sum(traj_list[0].is_last())
		episodes.assign_add(games_completed)
		if games_completed > 0:
			log_str = ""
			log_str += "[TRAIN]  "
			log_str += "[global: {}] ".format(str(train_global_step.numpy()).rjust(10))
			log_str += "[games: {}%] ".format(str(int(episodes.numpy() / total_episodes * 100)).rjust(5))
			log_str += "[step rate: {}] ".format(str(train_rate_tracker.steps_per_second())[:7])
			log_str += "[iter rate: {}] ".format(str(iteration_rate_tracker.steps_per_second())[:7])
			print(log_str)

	def summarize_with_writer(writer):
		def handle_summaries(traj_list):
			finished_games = tf.where(traj_list[0].is_last())
			observation, _ = traj_list[0].observation
			observation = tf.squeeze(tf.gather(observation, finished_games), axis=(1,))
			if observation.shape[0]:
				game_metrics, _ = metrics.summarize(observation)
				with tf.name_scope("summary"):
					with writer.as_default(step=train_global_step.numpy()):
						for game_metric in game_metrics:
							data = game_metrics[game_metric].numpy()
							for scalar in data:
								tf.summary.scalar(name=game_metric.lower(), data=scalar)
				with tf.name_scope("benchmark"):
					for turns in game_metrics[turn_number]:
						episode_turn_list.append(turns)
						tf.summary.scalar(name="turn_number", data=turns)
						tf.summary.scalar(name="mean", data=np.mean(episode_turn_list))
						tf.summary.scalar(name="standard_deviation", data=np.std(episode_turn_list))
		return handle_summaries

	metrics = ObservationMetrics()

	train_env = get_env(
		process_count=train_process_count,
		game_count=train_game_count)

	train_policy_mapping = random_policy_mapping(
		agent_count=agent_count,
		player_count=player_count,
		game_count=train_game_count,
		process_count=train_process_count)

	iteration_rate_tracker = StepPerSecondTracker(iteration)
	train_rate_tracker = StepPerSecondTracker(train_global_step)

	def splitter(observation):
		obs, mask = observation
		return obs, mask

	random_policy = RandomTFPolicy(
		time_step_spec=train_env.time_step_spec(),
		action_spec=train_env.action_spec(),
		observation_and_action_constraint_splitter=splitter)

	train_driver = MappedDriver(
		environment=train_env,
		policy_list=[random_policy] * agent_count,
		policy_mapping=train_policy_mapping,
		max_episodes=total_episodes,
		observers=[
			update_train_global_step,
			train_log,
			summarize_with_writer(summary_writer),
		])

	train_driver.run()


def main(_):
	random_benchmark()


if __name__ == '__main__':
	tf_agents.system.multiprocessing.handle_main(main)
