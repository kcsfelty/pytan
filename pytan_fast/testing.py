import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import TensorSpec
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.distributions import reparameterized_sampling
from tf_agents.environments import tf_py_environment
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.replay_buffers.episodic_replay_buffer import EpisodicReplayBuffer
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories import TimeStep, trajectory, PolicyStep
from tf_agents.utils import common

from pytan_fast.game import PyTanFast
from pytan_fast.settings import player_count

support = np.linspace(-10, 10, 51, dtype=np.float32)
neg_inf = tf.constant(-np.inf, dtype=tf.float32)
seed_stream = tfp.util.SeedStream(seed=None, salt='tf_agents_tf_policy')


@tf.function
def get_actions(obs_tuple):
	obs, mask = obs_tuple
	q_logits = categorical_q_net(obs)[0]
	q_values = common.convert_q_logits_to_values(q_logits, support)
	logits = tf.compat.v2.where(tf.cast(mask, tf.bool), q_values, neg_inf)
	dist = tfp.distributions.Categorical(logits=logits, dtype=tf.float32)
	return tf.nest.map_structure(
		lambda d: reparameterized_sampling.sample(d, seed=seed_stream()), dist)


@tf.function
def get_player_actions(time_step):
	obs_list, mask_list = time_step.observation
	obs_list = tf.reshape(obs_list, (player_count, game_count, -1))
	mask_list = tf.reshape(mask_list, (player_count, game_count, -1))
	actions = tf.map_fn(get_actions, (obs_list, mask_list), fn_output_signature=tf.TensorSpec((game_count,), dtype=tf.float32))
	actions = tf.cast(actions, dtype=tf.int32)
	return actions


def splitter(obs_tuple):
	obs, mask = obs_tuple
	return obs, mask


game_count = 1000
steps = 1000000
action_count = 379
observation_count = 1402
n_step_update = 10
batch_size = player_count * game_count

global_step = tf.Variable(0, dtype=tf.int32)
game = PyTanFast(game_count=game_count, global_step=global_step)
tf_game = tf_py_environment.TFPyEnvironment(game)

fake_action_spec = BoundedTensorSpec(shape=(), dtype=tf.int32, name='action_mask', minimum=np.array(0), maximum=np.array(action_count - 1))
step_type_spec = TensorSpec(shape=(), dtype=tf.int32, name='step_type')
reward_type_spec = BoundedTensorSpec(shape=(), dtype=tf.float32, name='reward', minimum=np.array(-1., dtype=np.float32), maximum=np.array(1., dtype=np.float32))
discount_type_spect = BoundedTensorSpec(shape=(), dtype=tf.float32, name='discount', minimum=np.array(0., dtype=np.float32), maximum=np.array(1., dtype=np.float32))
obs_type_spec = BoundedTensorSpec(shape=(observation_count,), dtype=tf.int32, name='observation', minimum=np.array(0), maximum=np.array(128))
mask_type_spec = BoundedTensorSpec(shape=(action_count,), dtype=tf.int32, name='action_mask', minimum=np.array(0), maximum=np.array(1))
fake_time_step_spec = TimeStep(step_type_spec, reward_type_spec, discount_type_spect, (obs_type_spec, mask_type_spec))

categorical_q_net = categorical_q_network.CategoricalQNetwork(
	input_tensor_spec=obs_type_spec,
	action_spec=fake_action_spec,
	fc_layer_params=(2**6, 2**6, 2**6, 2**6,))

train_counter = tf.Variable(0, dtype=tf.int32)

agent = categorical_dqn_agent.CategoricalDqnAgent(
	time_step_spec=fake_time_step_spec,
	action_spec=fake_action_spec,
	categorical_q_network=categorical_q_net,
	train_step_counter=train_counter,
	optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001),
	n_step_update=n_step_update,
	min_q_value=-1,
	max_q_value=1,
	observation_and_action_constraint_splitter=splitter)


time_step = tf_game.current_time_step()
start = time.perf_counter()

time_step_acc = 0
next_steps_acc = 0
trajectory_acc = 0
add_batch_acc = 0
total_acc = 0

replay_buffer = TFUniformReplayBuffer(
	data_spec=agent.collect_data_spec,
	batch_size=batch_size,
	max_length=1000
)

dataset = replay_buffer.as_dataset(
	num_parallel_calls=3,
	sample_batch_size=batch_size,
	num_steps=n_step_update + 1,
).prefetch(3)
iterator = iter(dataset)

log_interval = 300
last_step = 0
last_log = time.time()


for step in range(steps):
	mark = time.perf_counter()
	action = get_player_actions(time_step)
	next_steps_acc += time.perf_counter() - mark

	mark = time.perf_counter()
	next_time_step = tf_game.step(action)
	time_step_acc += time.perf_counter() - mark

	mark = time.perf_counter()
	action_flat = tf.reshape(action, (1, player_count * game_count))[0]
	traj = trajectory.from_transition(
		time_step,
		PolicyStep(action_flat),
		next_time_step
	)
	time_step = next_time_step
	trajectory_acc += time.perf_counter() - mark

	mark = time.perf_counter()
	episode_ids = replay_buffer.add_batch(traj)
	add_batch_acc += time.perf_counter() - mark

	if replay_buffer.num_frames() > batch_size * n_step_update:
		exp, _ = next(iterator)
		agent.train(exp)

	now = time.time()
	if now > last_log + log_interval:
		current_step = step * game_count
		delta = (current_step - last_step) / (now - last_log)

		last_step = current_step
		last_log = now

		print(current_step, "of", steps * game_count, str(int(step / steps * 100)) + "%", "rate", delta, "train:", train_counter.numpy().item())
		next_log = time.time() + log_interval
		total_acc = time_step_acc + next_steps_acc + add_batch_acc
		# print("time_step_acc", str(int(time_step_acc / total_acc * 100)) + "%")
		# print("next_steps_acc", str(int(next_steps_acc / total_acc * 100)) + "%")
		# print("trajectory_acc", str(int(trajectory_acc / total_acc * 100)) + "%")
		# print("add_batch_acc", str(int(add_batch_acc / total_acc * 100)) + "%")
		# print("train_counter", train_counter)

# print(cache._completed_episodes())
# print(cache._get_episode(episode_ids[0]))

end = time.perf_counter()
delta = end - start
print()
print("duration", delta)
print("steps", steps * game_count)
print("rate", steps * game_count / delta, "steps/s")

print()
print()
print("time_step_acc", time_step_acc)
print("next_steps_acc", next_steps_acc)
print("add_batch_acc", add_batch_acc)





















