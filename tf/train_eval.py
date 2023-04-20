import math
import time
from abc import ABC

import tensorflow as tf
from tf_agents.environments import tf_py_environment, ParallelPyEnvironment

from tf.agent import Agent
from game.game import PyTanFast
from reference.settings import player_count


def train_eval(
		# Performance
		process_count=None,
		thread_count=2 ** 5,

		# Batching
		game_count=2 ** 8,
		total_steps=500e6,
		initial_steps=3e6,
		n_step_update=2 ** 7,

		# Replay buffer
		replay_buffer_size=2 ** 12,
		replay_batch_size=2 ** 6,

		# Network parameters
		learn_rate=1e-4,
		fc_layer_params=(2 ** 10, 2 ** 9, 2 ** 8,),

		# Greedy policy epsilon
		epsilon_greedy_start=1.0,
		epsilon_greedy_end=0.1,
		epsilon_greedy_half_life=10e6,

		# Intervals
		train_interval=2 ** 5,
		eval_interval=2 ** 14,
		log_interval=2 ** 7,
	):
	global_step = tf.Variable(0, dtype=tf.int32)
	epsilon_greedy_delta = tf.constant(epsilon_greedy_start - epsilon_greedy_end)
	epsilon_greedy_base = tf.constant(1 - math.log(2) / epsilon_greedy_half_life)

	def epsilon_greedy():
		epsilon = tf.math.pow(epsilon_greedy_base, tf.cast(global_step, tf.float32))
		epsilon = tf.multiply(epsilon, epsilon_greedy_delta)
		epsilon = tf.add(epsilon_greedy_end, epsilon)
		return epsilon

	def log():
		step = global_step.numpy().item()
		log_str = ""
		log_str += "[global: {}] ".format(str(step).rjust(10))
		log_str += "[iteration: {}] ".format(str(iteration).rjust(5))
		log_str += "[pct: {}%] ".format(str(int(step / total_steps * 100)))
		log_str += "[rate: {} step/sec] ".format(str(int(rate)).rjust(5))
		log_str += "[epsilon: {}] ".format(str(epsilon_greedy().numpy().item())[:7])
		print(log_str)

	def get_rate():
		step = global_step.numpy().item()
		step_delta = step - last_step
		time_delta = time.perf_counter() - last_time
		return step, time.perf_counter(), step_delta / time_delta

	def act(time_step_list):
		action_list = []
		for agent_, time_step_ in zip(agent_list, time_step_list):
			agent_.add_batch(time_step_)
			agent_.action = agent_.agent.collect_policy.action(time_step_)
			action_list.append(agent_.action.action)
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

	def get_agent_list():
		return [Agent(
			index=index,
			action_spec=env.action_spec(),
			time_step_spec=env.time_step_spec(),
			game_count=game_count,
			replay_buffer_size=replay_buffer_size,
			replay_batch_size=replay_batch_size,
			fc_layer_params=fc_layer_params,
			n_step_update=n_step_update,
			learn_rate=learn_rate,
			epsilon_greedy=epsilon_greedy,
		) for index in range(player_count)]

	last_time = time.perf_counter()
	last_step = global_step.numpy().item()
	iteration = 0

	env = get_env()
	agent_list = get_agent_list()
	time_step = env.current_time_step()

	while global_step.numpy() < initial_steps:
		action = act(time_step)
		time_step = env.step(action)
		if iteration % log_interval == 0:
			last_step, last_time, rate = get_rate()
			log()
		iteration += 1

	print("Initial steps completed. Beginning Training.")

	while global_step.numpy() < total_steps:
		action = act(time_step)
		time_step = env.step(action)
		if iteration % train_interval == 0:
			for agent in agent_list:
				agent.train()
		if iteration % eval_interval == 0:
			pass
		if iteration % log_interval == 0:
			last_step, last_time, rate = get_rate()
			log()
		iteration += 1


if __name__ == "__main__":
	train_eval()
