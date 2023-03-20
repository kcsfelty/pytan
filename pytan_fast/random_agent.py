import tensorflow as tf
from tf_agents.policies.random_tf_policy import RandomTFPolicy


def splitter(obs_tuple):
	obs, mask = obs_tuple
	return obs, mask


class RandomAgent:
	def __init__(self, env_specs, player_index, log_dir, observers=None):
		self.env_specs = env_specs
		self.player_index = player_index
		self.log_dir = log_dir
		self.writer = tf.compat.v2.summary.create_file_writer(log_dir + "/agent" + str(self.player_index))
		self.observers = observers or []
		self.policy = RandomTFPolicy(
			time_step_spec=env_specs["env_time_step_spec"],
			action_spec=env_specs["env_action_spec"],
			observation_and_action_constraint_splitter=splitter)

	def get_policy(self, collect=True):
		return self.policy

	def write_summary(self, summaries, step):
		with self.writer.as_default():
			for summary_key in summaries["scalars"]:
				tf.summary.scalar(
					name=summary_key,
					data=summaries["scalars"][summary_key],
					step=step.item())
			for summary_key in summaries["histograms"]:
				tf.summary.histogram(
					name=summary_key,
					data=summaries["histograms"][summary_key],
					step=step.item(),
					buckets=len(summaries["histograms"][summary_key]))