import math
from abc import ABC

import tf_agents
from tf_agents.environments import tf_py_environment, ParallelPyEnvironment

from game.game import PyTan
from reference.settings import player_count
from tf.ObservationMetrics import ObservationMetrics
from tf.agent import Agent
from tf.mapped_driver import MappedDriver
from tf.random_policy_mapping import random_policy_mapping
import tensorflow as tf

goal_episode_steps = 1200
goal_player_steps = goal_episode_steps // player_count
oldest_n_step_discount_factor = 3
half_life_steps = goal_player_steps / oldest_n_step_discount_factor
n_step_gamma = 1 - math.log(2) / half_life_steps


def mapped_train_eval(
		# Performance / Logging
		log_dir="./logs/current",
		train_process_count=6,
		eval_process_count=1,
		thread_count=2 ** 5,
		agent_count=3,

		# Batching
		train_game_count=2 ** 6,
		eval_game_count=2 ** 4,
		n_step_update=goal_player_steps,

		# Replay buffer
		replay_buffer_size=goal_player_steps + 1,
		replay_batch_size=2 ** 3,

		# Network parameters
		learn_rate=1e-5,
		fc_layer_params=(2 ** 6, 2 ** 5,),
		gamma=n_step_gamma,

		# Greedy policy epsilon
		epsilon_greedy_start=1.00,
		epsilon_greedy_end=0.01,
		epsilon_greedy_half_life=10e6,

		# Intervals
		total_steps=500e6,
		initial_steps=1e6,
		eval_steps=75000,
		train_steps=2 ** 3,
		train_per_eval=1000,
	):
	train_global_step = tf.Variable(0, dtype=tf.int32)
	eval_global_step = tf.Variable(0, dtype=tf.int32)
	epsilon_greedy_delta = tf.constant(epsilon_greedy_start - epsilon_greedy_end)
	epsilon_greedy_base = tf.constant(1 - math.log(2) / epsilon_greedy_half_life)

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
			game_count=train_game_count,
			replay_buffer_size=replay_buffer_size,
			replay_batch_size=replay_batch_size,
			fc_layer_params=fc_layer_params,
			n_step_update=n_step_update,
			learn_rate=learn_rate,
			epsilon_greedy=epsilon_greedy,
			gamma=gamma,
			log_dir=log_dir + "/agents"
		) for index in range(agent_count)]

	def handle_summaries(traj_list):
		for agent, traj in zip(agent_list, traj_list):
			observation, _ = traj.observation
			print(metrics.summarize(observation))

	def increment_global_step(traj_list):
		pass

	def handle_trajectory_list(traj_list):
		for agent, traj in zip(agent_list, traj_list):
			agent.replay_buffer.add_batch(traj)

	def log():
		log_str = ""
		log_str += "[EVAL]  "
		log_str += "[global: {}] ".format(str(train_global_step).rjust(10))
		log_str += "[grads: {}] ".format(str(gradient_update_count).rjust(6))
		log_str += "[pct: {}%] ".format(str(int(train_global_step / train_steps * 100)).rjust(5))
		log_str += "[epsilon: {}] ".format(str(epsilon_greedy().numpy().item())[:7])

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

	train_driver = MappedDriver(
		environment=train_env,
		policy_list=[agent.agent.collect_policy for agent in agent_list],
		policy_mapping=train_policy_mapping,
		max_steps=train_steps,
		observers=[
			handle_trajectory_list,
			handle_summaries])

	eval_driver = MappedDriver(
		environment=eval_env,
		policy_list=[agent.agent.policy for agent in agent_list],
		policy_mapping=eval_policy_mapping,
		max_steps=eval_steps,
		observers=[])

	print("Performing evaluation.")
	eval_driver.run()

	while train_global_step.numpy() < initial_steps:
		train_driver.run()

	print("Initial steps completed. Beginning Training.")
	gradient_update_count = 0
	while train_global_step.numpy() < total_steps:
		log()
		for _ in range(train_per_eval):
			train_driver.run()
			gradient_update_count += 1
			for agent in agent_list:
				agent.train()
		print("Performing evaluation.")
		eval_env.reset()
		eval_driver.run()


if __name__ == '__main__':
	tf_agents.system.multiprocessing.handle_main(mapped_train_eval)
