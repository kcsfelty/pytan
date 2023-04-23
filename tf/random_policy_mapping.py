import tensorflow as tf


def random_policy_mapping(agent_count=1, player_count=1, game_count=1, process_count=1):
	batch_count = player_count * game_count * process_count
	mapping = tf.range(batch_count, dtype=tf.int32)
	mapping = tf.random.shuffle(mapping)
	mapping = tf.reshape(mapping, (agent_count, -1))
	return mapping
