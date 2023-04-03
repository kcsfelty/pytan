import os
from abc import ABC
from threading import Timer
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)
from tf_agents.environments import tf_py_environment
import time
from pytan_fast.agent import FastAgent
from pytan_fast.game import PyTanFast
from pytan_fast.settings import player_count
import multiprocessing
from multiprocessing.managers import BaseManager


class GlobalStep:

	def __init__(self):
		self.value = tf.Variable(0, trainable=False, dtype=tf.int64)

	def assign_add(self, value):
		self.value.assign_add(value)

	def read_value(self):
		return self.value.read_value()

	def numpy(self):
		return self.value.numpy()


class PyTanManager(BaseManager):
	pass


agent_names = ["FastAgent" + str(player_index) for player_index in range(player_count)]

for _agent_name in agent_names:
	PyTanManager.register(_agent_name, FastAgent)

PyTanManager.register("GlobalStep", GlobalStep)


def do_game(agent_list, env_id, global_step, total_steps):
	print("starting game on ", env_id, multiprocessing.current_process().name)
	mp_game = PyTanFast(agent_list, env_index=env_id, global_step=global_step, lock=lock)
	mp_game.reset()
	while global_step.read_value() < total_steps:
		mp_game.walk()
	print("game finished", env_id, multiprocessing.current_process().name)


def get_manager():
	m = PyTanManager()
	m.start()
	return m

def async_raise(error):
	raise error


def init(_lock):
	global lock
	lock = _lock


def train_eval(
	env_specs,
	env_count=1,
	log_dir="./logs/",
	total_steps=100000000,

	# Training / Experience,
	replay_ratio=10,
	replay_buffer_capacity=500000,

	# Hyperparameters,
	fc_layer_params=(2**7, 2**6),
	learning_rate=0.0001,
	n_step_update=50,
	eps_decay_rate=1 - np.log(2) / 200000,
	min_train_frames=20000,

	# Intervals,
	eval_interval=1000,
	train_interval=40,
	checkpoint_interval=10000,
	):
	manager = get_manager()
	global_step = manager.GlobalStep()
	agent_list = [manager[agent_name](
		player_index=player_index,
		batch_size=train_interval * replay_ratio,
		log_dir=log_dir,
		replay_buffer_capacity=int(replay_buffer_capacity/env_count),
		fc_layer_params=fc_layer_params,
		learning_rate=learning_rate,
		n_step_update=n_step_update,
		env_specs=env_specs,
		train_interval=train_interval,
		checkpoint_interval=checkpoint_interval,
		eval_interval=eval_interval,
		eps_decay_rate=eps_decay_rate,
		min_train_frames=min_train_frames,
		env_count=env_count
	) for player_index, agent_name in enumerate(agent_names)]

	def logging(wait=300):
		start = time.perf_counter()
		last_step = int(global_step.read_value())
		while last_step < total_steps:
			this_step = int(global_step.read_value())
			rate = this_step / (time.perf_counter() - start)
			print("Current Step:", this_step, "rate:", rate, "steps/s")
			last_step = this_step
			time.sleep(wait)

	log_timer = Timer(1, logging)
	log_timer.start()

	lock = multiprocessing.Lock()
	pool = multiprocessing.Pool(env_count, initializer=init, initargs=(lock,))
	for i in range(env_count):
		pool.apply_async(
			func=do_game,
			args=(agent_list, i, global_step, total_steps),
			error_callback=async_raise,
		)
	pool.close()
	pool.join()
	log_timer.cancel()
	print("finished")


if __name__ == "__main__":
	_env = PyTanFast()
	train_env = tf_py_environment.TFPyEnvironment(_env)
	_env_specs = {
		"env_observation_spec": train_env.observation_spec()[0],
		"env_action_spec": train_env.action_spec()[0],
		"env_time_step_spec": train_env.time_step_spec(),
	}

	policy_half_life_steps = 5000
	decay_rate = np.log(2) / policy_half_life_steps

	train_eval(
		env_specs=_env_specs,
		learning_rate=decay_rate,
		eval_interval=policy_half_life_steps * 7,
		replay_buffer_capacity=500000,
		replay_ratio=10,
		# env_count=multiprocessing.cpu_count(),
		env_count=4,
	)
