import math
import os.path
from abc import ABC

import tf_agents
from tf_agents.environments import tf_py_environment, ParallelPyEnvironment
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
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
		log_dir=os.path.join("./logs", "random"),
		train_process_count=7,
		thread_count=2 ** 5,
		agent_count=1,

		# Batching
		train_game_count=2 ** 8,

		# Intervals
		total_steps=500e6,
	):
	iteration = tf.Variable(0, dtype=tf.int32)
	train_global_step = tf.Variable(0, dtype=tf.int32)

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

	def update_train_global_step(traj_list):
		existing_games = len(tf.where(~traj_list[0].is_first()))
		train_global_step.assign_add(existing_games)
		iteration.assign_add(1)

	def train_log():
		if iteration.numpy() % 2 ** 4 == 0:
			log_str = ""
			log_str += "[TRAIN]  "
			log_str += "[global: {}] ".format(str(train_global_step.numpy()).rjust(10))
			log_str += "[pct: {}%] ".format(str(int(train_global_step.numpy() / total_steps * 100)).rjust(5))
			log_str += "[step rate: {}] ".format(str(train_rate_tracker.steps_per_second())[:7])
			log_str += "[iter rate: {}] ".format(str(iteration_rate_tracker.steps_per_second())[:7])
			print(log_str)

	train_env = get_env(
		process_count=train_process_count,
		game_count=train_game_count,
		env_id_prefix="train")

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
		max_steps=total_steps,
		observers=[
			update_train_global_step,
			lambda _: train_log(),
		])

	train_driver.run()


def main(_):
	mapped_train_eval()


if __name__ == '__main__':
	tf_agents.system.multiprocessing.handle_main(main)
