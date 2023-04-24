import math
import time
from abc import ABC

import tensorflow as tf
from tf_agents.environments import tf_py_environment, ParallelPyEnvironment

from tf.agent import Agent
from game.game import PyTan
from reference.settings import player_count

goal_episode_steps = 1200
goal_player_steps = goal_episode_steps // player_count
oldest_n_step_discount_factor = 3
half_life_steps = goal_player_steps / oldest_n_step_discount_factor
n_step_gamma = 1 - math.log(2) / half_life_steps


def train_eval(
		# Performance / Logging
		log_dir="./logs/current",
		train_process_count=None,
		eval_process_count=None,
		thread_count=2 ** 5,

		# Batching
		train_game_count=2 ** 10,
		eval_game_count=2 ** 4,
		n_step_update=goal_player_steps,

		# Replay buffer
		replay_buffer_size=goal_player_steps + 1,
		replay_batch_size=2 ** 3,

		# Network parameters
		learn_rate=1e-5,
		fc_layer_params=(2 ** 7, 2 ** 6,),
		gamma=n_step_gamma,

		# Greedy policy epsilon
		epsilon_start=1.00,
		epsilon_end=0.01,
		epsilon_half_life=10e6,

		# Intervals
		total_steps=500e6,
		initial_steps=10e6,
		eval_steps=75000,
		train_interval=2 ** 3,
		eval_interval=2 ** 9,
		log_interval=2 ** 6,
	):
	train_global_step = tf.Variable(0, dtype=tf.int32)
	eval_global_step = tf.Variable(0, dtype=tf.int32)
	epsilon_greedy_delta = tf.constant(epsilon_start - epsilon_end)
	epsilon_greedy_base = tf.constant(1 - math.log(2) / epsilon_half_life)

	def epsilon_greedy():
		epsilon = tf.math.pow(epsilon_greedy_base, tf.cast(train_global_step, tf.float32))
		epsilon = tf.multiply(epsilon, epsilon_greedy_delta)
		epsilon = tf.add(epsilon_end, epsilon)
		return epsilon

	def log():
		step = train_global_step.numpy().item()
		log_str = ""
		log_str += "[TRAIN] "
		log_str += "[global: {}] ".format(str(step).rjust(10))
		log_str += "[iteration: {}] ".format(str(iteration).rjust(5))
		log_str += "[pct: {}%] ".format(str(int(step / total_steps * 100)).rjust(5))
		log_str += "[rate: {} step/sec] ".format(str(int(rate)).rjust(5))
		log_str += "[epsilon: {}] ".format(str(epsilon_greedy().numpy().item())[:7])
		print(log_str)

	def get_rate():
		step = train_global_step.numpy().item()
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
		eval_end_step = eval_global_step.numpy() + eval_steps
		eval_last_step = eval_global_step.numpy().item()
		eval_iteration = 0
		eval_time_step = eval_env.reset()
		while eval_global_step.numpy() < eval_end_step:
			eval_action_list = []
			for agent_, time_step_ in zip(agent_list, eval_time_step):
				eval_action_list.append(agent_.agent.policy.action(time_step_).action)
			eval_action = tf.convert_to_tensor(eval_action_list, dtype=tf.int32)
			eval_time_step = eval_env.step(eval_action)
			if eval_iteration % log_interval == 0:
				eval_step = eval_global_step.numpy().item()
				eval_step_delta = eval_step - eval_last_step
				eval_time_delta = time.perf_counter() - eval_last_time
				eval_rate = eval_step_delta / eval_time_delta
				eval_last_time = time.perf_counter()
				eval_last_step = eval_step
				log_str = ""
				log_str += "[EVAL]  "
				log_str += "[global: {}] ".format(str(eval_step).rjust(10))
				log_str += "[iteration: {}] ".format(str(eval_iteration).rjust(5))
				log_str += "[pct: {}%] ".format(str(int(1 - (eval_end_step - eval_global_step.numpy()) / eval_steps * 100)).rjust(5))
				log_str += "[rate: {} step/sec] ".format(str(int(eval_rate)).rjust(5))
				print(log_str)
			eval_iteration += 1

	def get_env(process_count=None, as_eval_env=False):
		game_count_ = train_game_count if not as_eval_env else eval_game_count
		global_step_ = train_global_step if not as_eval_env else eval_global_step
		env_id_prefix = "train" if not as_eval_env else "eval"
		log_dir_ = log_dir + "/" + env_id_prefix

		if process_count:
			meta_env_list = []
			for i in range(process_count):
				class ParallelPyTan(PyTan, ABC):
					def __init__(self):
						super().__init__(
							game_count=game_count_,
							global_step=global_step_,
							worker_count=thread_count,
							log_dir=log_dir_,
							env_id="{}-{}".format(env_id_prefix, i))
				meta_env_list.append(ParallelPyTan)
			py_env = ParallelPyEnvironment(meta_env_list)
		else:
			py_env = PyTan(
				game_count_,
				global_step_,
				log_dir=log_dir_,
				env_id=env_id_prefix)

		return tf_py_environment.TFPyEnvironment(py_env)

	def get_agent_list():
		return [Agent(
			index=index,
			action_spec=train_env.action_spec(),
			time_step_spec=train_env.time_step_spec(),
			game_count=train_game_count,
			replay_buffer_size=replay_buffer_size,
			replay_batch_size=replay_batch_size,
			fc_layer_params=fc_layer_params,
			n_step_update=n_step_update,
			learn_rate=learn_rate,
			epsilon_greedy=epsilon_greedy,
			gamma=gamma,
			log_dir=log_dir + "/agents"
		) for index in range(player_count)]

	last_time = time.perf_counter()
	last_step = train_global_step.numpy().item()
	iteration = 0

	train_env = get_env(train_process_count)
	eval_env = get_env(eval_process_count, as_eval_env=True)
	agent_list = get_agent_list()
	train_time_step = train_env.current_time_step()

	print("Performing initial evaluation.")
	run_eval()

	while iteration < replay_buffer_size or train_global_step.numpy() < initial_steps:
		action = act(train_time_step)
		train_time_step = train_env.step(action)
		if iteration % log_interval == 0:
			last_step, last_time, rate = get_rate()
			log()
		iteration += 1

	print("Initial steps completed. Beginning Training.")

	while train_global_step.numpy() < total_steps:
		action = act(train_time_step)
		train_time_step = train_env.step(action)
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
