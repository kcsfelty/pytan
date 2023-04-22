import math
import time
from abc import ABC

import tensorflow as tf
from tf_agents.environments import tf_py_environment, ParallelPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from tf.agent import Agent
from game.game import PyTanFast
from reference.settings import player_count


def train_eval(
		# Performance
		process_count=None,
		thread_count=2 ** 5,

		# Batching
		game_count=2 ** 8,
		total_steps=250e6,
		log_interval=2 ** 7,
	):
	global_step = tf.Variable(0, dtype=tf.int32)

	def log():
		step = global_step.numpy().item()
		log_str = ""
		log_str += "[global: {}] ".format(str(step).rjust(10))
		log_str += "[iteration: {}] ".format(str(iteration).rjust(5))
		log_str += "[pct: {}%] ".format(str(int(step / total_steps * 100)))
		log_str += "[rate: {} step/sec] ".format(str(int(rate)).rjust(5))
		print(log_str)

	def get_rate():
		step = global_step.numpy().item()
		step_delta = step - last_step
		time_delta = time.perf_counter() - last_time
		return step, time.perf_counter(), step_delta / time_delta

	def act(time_step_list):
		action_list = []
		for policy_, time_step_ in zip(policy_list, time_step_list):
			action_list.append(policy_.action(time_step_))
		return tf.convert_to_tensor(action_list, dtype=tf.int32)

	def get_env():
		if process_count:
			class ParallelPyTan(PyTanFast, ABC):
				def __init__(self):
					super().__init__(
						game_count=game_count,
						global_step=global_step,
						worker_count=thread_count)
			env_list = [ParallelPyTan] * process_count
			py_env = ParallelPyEnvironment(env_list)
		else:
			py_env = PyTanFast(game_count, global_step)

		return tf_py_environment.TFPyEnvironment(py_env)

	def splitter(observation):
		obs, mask = observation
		return obs, mask

	def get_policy_list():
		return [RandomTFPolicy(
			action_spec=env.action_spec(),
			time_step_spec=env.time_step_spec(),
			observation_and_action_constraint_splitter=splitter,
		) for _ in range(player_count)]

	last_time = time.perf_counter()
	last_step = global_step.numpy().item()
	iteration = 0

	env = get_env()
	policy_list = get_policy_list()
	time_step = env.current_time_step()

	while global_step.numpy() < total_steps:
		action = act(time_step)
		time_step = env.step(action)
		if iteration % log_interval == 0:
			last_step, last_time, rate = get_rate()
			log()
		iteration += 1


if __name__ == "__main__":
	train_eval()
