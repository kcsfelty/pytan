from game import game_settings
import tensorflow as tf

global_step = tf.compat.v1.train.get_or_create_global_step()


# Main Hyperparmeters
total_steps = 20000000
starting_iteration = 0
num_iterations = 200
turns_per_player = 33
epsilon_start = 0.1
epsilon_end = 0.0001
epsilon_steps = num_iterations // 2
epsilon_delta = (epsilon_start - epsilon_end) / epsilon_steps
gamma = 1 - (turns_per_player / 2) ** -1
reward_victory_bonus = 3
epsilon_list = [epsilon_start - epsilon_delta * i for i in range(starting_iteration, epsilon_steps)]

# Environment
turn_limit = turns_per_player * game_settings.player_count
actions_per_turn = 17
steps_per_episode = turn_limit * actions_per_turn

# Network
learning_rate = 1e-3
fc_layer_params = (2 ** 12, 2 ** 11, 2 ** 10)

# Training
episodes_per_iteration = total_steps // steps_per_episode // num_iterations
iteration_in_replay_buffer = 3
training_batch_size = 512
train_steps_per_iteration = (steps_per_episode * episodes_per_iteration) // training_batch_size
training_num_steps = 2
replay_buffer_size = steps_per_episode * episodes_per_iteration * iteration_in_replay_buffer
initial_collect_episodes = episodes_per_iteration
action_drain = -1 * 4 * reward_victory_bonus / steps_per_episode
total_games_to_play = total_steps // steps_per_episode


# Rewards
victory_points_min = 2
victory_points_max = game_settings.victory_points_to_win
victory_points_range = victory_points_max - victory_points_min

reward_spread = 2

min_reward = -1 * reward_spread
max_reward = 1 * reward_spread


# Logging
plot_interval = 1
render_frequency = 10
