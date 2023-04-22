import math
import time
from abc import ABC

import tensorflow as tf
from tf_agents.environments import tf_py_environment, ParallelPyEnvironment

from tf.agent import Agent
from game.game import PyTanFast
from reference.settings import player_count

goal_episode_steps = 1200
goal_player_steps = goal_episode_steps // player_count
oldest_n_step_discount_factor = 3
half_life_steps = goal_player_steps / oldest_n_step_discount_factor
n_step_gamma = 1 - math.log(2) / half_life_steps


def train_eval(
		# Performance / Logging
		log_dir="./logs",
		process_count=None,
		thread_count=2 ** 5,

		# Batching
		game_count=2 ** 10,
		eval_game_count=2**4,
		n_step_update=goal_player_steps,

		# Replay buffer
		replay_buffer_size=goal_player_steps,
		replay_batch_size=2 ** 3,

		# Network parameters
		learn_rate=1e-4,
		fc_layer_params=(2 ** 6, 2 ** 5,),
		gamma=n_step_gamma,

		# Greedy policy epsilon
		epsilon_greedy_start=1.00,
		epsilon_greedy_end=0.01,
		epsilon_greedy_half_life=10e6,

		# Intervals
		total_steps=500e6,
		eval_steps=50000,
		train_interval=2 ** 3,
		eval_interval=2 ** 14,
		log_interval=2 ** 7,
	):
	global_step = tf.Variable(0, dtype=tf.int32)
	eval_global_step = tf.Variable(0, dtype=tf.int32)
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

	def run_eval():
		eval_last_time = time.perf_counter()
		eval_last_step = global_step.numpy().item()
		eval_iteration = 0
		eval_time_step = eval_env.reset()
		while eval_global_step.numpy() < eval_steps:
			eval_action_list = []
			for agent_, time_step_ in zip(agent_list, eval_time_step):
				eval_action_list.append(agent_.agent.policy.action(time_step_))
			eval_action = tf.convert_to_tensor(eval_action_list, dtype=tf.int32)
			eval_time_step = train_env.step(eval_action)
			if iteration % log_interval == 0:
				eval_step = eval_global_step.numpy().item()
				eval_step_delta = eval_step - eval_last_step
				eval_time_delta = time.perf_counter() - eval_last_time
				eval_rate = eval_step_delta / eval_time_delta
				eval_last_time = time.perf_counter()
				eval_last_step = eval_step
				log_str = ""
				log_str += "[global: {}] ".format(str(eval_step).rjust(10))
				log_str += "[iteration: {}] ".format(str(eval_iteration).rjust(5))
				log_str += "[pct: {}%] ".format(str(int(eval_step / eval_steps * 100)))
				log_str += "[rate: {} step/sec] ".format(str(int(eval_rate)).rjust(5))
				print(log_str)

	def get_env(as_eval_env=False):
		game_count_ = game_count if not as_eval_env else eval_game_count,
		global_step_ = global_step if not as_eval_env else eval_global_step,
		log_dir_ = log_dir if not eval_env else log_dir + "/eval"
		if process_count:
			class ParallelPyTan(PyTanFast, ABC):
				def __init__(self):
					super().__init__(
						game_count=game_count_,
						global_step=global_step_,
						worker_count=thread_count,
						log_dir=log_dir_)
			env_list = [ParallelPyTan] * process_count
			py_env = ParallelPyEnvironment(env_list)
		else:
			py_env = PyTanFast(
				game_count_,
				global_step_,
				log_dir=log_dir_)

		return tf_py_environment.TFPyEnvironment(py_env)

	def get_agent_list():
		return [Agent(
			index=index,
			action_spec=train_env.action_spec(),
			time_step_spec=train_env.time_step_spec(),
			game_count=game_count,
			replay_buffer_size=replay_buffer_size,
			replay_batch_size=replay_batch_size,
			fc_layer_params=fc_layer_params,
			n_step_update=n_step_update,
			learn_rate=learn_rate,
			epsilon_greedy=epsilon_greedy,
			gamma=gamma,
		) for index in range(player_count)]

	last_time = time.perf_counter()
	last_step = global_step.numpy().item()
	iteration = 0

	train_env = get_env()
	eval_env = get_env(as_eval_env=True)
	agent_list = get_agent_list()
	time_step = train_env.current_time_step()

	for _ in range(replay_buffer_size):
		action = act(time_step)
		time_step = train_env.step(action)
		if iteration % log_interval == 0:
			last_step, last_time, rate = get_rate()
			log()
		iteration += 1

	print("Initial steps completed. Beginning Training.")

	while global_step.numpy() < total_steps:
		action = act(time_step)
		time_step = train_env.step(action)
		if iteration % train_interval == 0:
			for agent in agent_list:
				agent.train()
		if iteration % eval_interval == 0:
			run_eval()
		if iteration % log_interval == 0:
			last_step, last_time, rate = get_rate()
			log()
		iteration += 1


if __name__ == "__main__":
	train_eval()
