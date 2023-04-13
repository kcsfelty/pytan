import time

from tf_agents.environments import tf_py_environment
import tensorflow as tf
from pytan_fast.agent import Agent
from pytan_fast.game import PyTanFast
from pytan_fast.meta_agent import MetaAgent
from pytan_fast.settings import player_count


def train_eval(
		game_count=1000,
		total_steps=250e6,
		train_interval=1,
		eval_interval=1,
		log_interval=100,
	):

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

	global_step = tf.Variable(0, dtype=tf.int32)
	game = PyTanFast(game_count, global_step)
	env = tf_py_environment.TFPyEnvironment(game)
	agent_list = [Agent(
		game_count=game_count,
		replay_buffer_size=1000,
		replay_batch_size=1000) for _ in range(player_count)]
	meta = MetaAgent(agent_list, game_count)
	start = time.perf_counter()
	run()


if __name__ == "__main__":
	train_eval()
