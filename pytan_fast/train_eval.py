import time
from abc import ABC
from tf_agents.environments import tf_py_environment, ParallelPyEnvironment
import tensorflow as tf
from pytan_fast.agent import Agent
from pytan_fast.game import PyTanFast
from pytan_fast.meta_agent import MetaAgent
from pytan_fast.settings import player_count


def train_eval(
		process_count=None,
		thread_count=32,
		game_count=250,
		total_steps=1e9,
		train_interval=1,
		eval_interval=1,
		log_interval=100,
		replay_buffer_size=10000,
		replay_batch_size=64,
		learn_rate=0.001,
		fc_layer_params=(2**10, 2**10,),
		n_step_update=1000,
	):
	global_step = tf.Variable(0, dtype=tf.int32)

	def maybe_train():
		if global_step.numpy() % train_interval == 0:
			meta.train()

	def maybe_eval():
		if global_step.numpy() % eval_interval == 0:
			pass

	def maybe_log():
		if global_step.numpy() % log_interval == 0:
			step = global_step.numpy().item()
			log_str = ""
			log_str += "[global: {}]".format(str(step).rjust(10))
			log_str += "\t"
			log_str += "[pct: {}%]".format(str(int(step / total_steps * 100)))
			log_str += "\t"
			log_str += "[rate: {} step/sec]".format(str(int(step / (time.perf_counter() - start))))
			print(log_str)

	def run():
		time_step = env.current_time_step()
		while global_step.numpy() < total_steps:
			action = meta.act(time_step)
			time_step = env.step(action)
			maybe_train()
			maybe_eval()
			maybe_log()

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
		game_count=game_count,
		replay_buffer_size=replay_buffer_size,
		replay_batch_size=replay_batch_size,
		fc_layer_params=fc_layer_params,
		n_step_update=n_step_update,
		learn_rate=learn_rate
	) for index in range(player_count)]

	env = get_env()
	agent_list = get_agent_list()
	meta = MetaAgent(agent_list, game_count)
	start = time.perf_counter()
	run()


if __name__ == "__main__":
	train_eval()
