import random
import time

import gym
import numpy as np
from catanatron import Game, RandomPlayer, Color, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.envs.catanatron_env import to_action_space
from catanatron_gym.features import create_sample_vector, create_sample

actions_count = 289 + 1

# @register_player("FOO")
class FooPlayer(Player):
  def decide(self, game, playable_actions, env):
    """Should return one of the playable_actions.

    Args:
        game (Game): complete game state. read-only.
        playable_actions (Iterable[Action]): options to choose from
    Return:
        action (Action): Chosen element of playable_actions
    """
    # ===== YOUR CODE HERE =====
    # As an example we simply return the first action:

    # sample = create_sample(game, self.color)
    # vector = create_sample_vector(game, self.color)
    # print(len(sample.keys()), len(vector))
    # for key, val in zip(sample.keys(), vector):
    #     print(key, val)
    mask = np.zeros(actions_count)
    mask[[to_action_space(action) for action in playable_actions]] = 1
    print(mask)
    return playable_actions[0]
    # ===== END YOUR CODE =====


# Play a simple 4v4 game
players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    FooPlayer(Color.WHITE)
    # RandomPlayer(Color.WHITE),
    # RandomPlayer(Color.ORANGE),
]

start = time.perf_counter()
for _ in range(900):

    game = Game(players)
    # print(game)
    print(game.play())  # returns winning color
end = time.perf_counter()
print(end - start)




# 3-player catan on a "Mini" map (7 tiles) until 6 points.
env = gym.make(
    "catanatron_gym:catanatron-v0",
    config={
        "map_type": "MINI",
        "vps_to_win": 6,
        "enemies": [FooPlayer(Color.RED), FooPlayer(Color.ORANGE)],
        "reward_function": my_reward_function,
        "representation": "mixed",
    },
)

# env = gym.make("catanatron_gym:catanatron-v0")
observation = env._reset()
# print(env.representation)
# print(env.observation_space["board"])
# print(env.observation_space["numeric"])
# for feature_extractor in feature_extractors:
#     print(feature_extractor(env, Color.WHITE))
# for _ in range(1000):
#   action = random.choice(env.get_valid_actions()) # your agent here (this takes random actions)
#
#   observation, reward, done, info = env.step(action)
#   print(observation["board"].shape)
#   print(observation["numeric"].shape)
#   print(reward)
#   print(done)
#   print(info)
#   if done:
#       observation = env.reset()
env.close()