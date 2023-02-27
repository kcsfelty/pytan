import unittest
import numpy as np
import pytan_fast.settings as gs
from tf_agents.trajectories import PolicyStep
from pytan_fast.game import PyTanFast


class Policy:
	def __init__(self):
		pass

	def action(self, time_step):
		return PolicyStep(action=0)


def traj_for(player_index):
	def add_traj(traj):
		print("added a traj for", player_index, traj)
	return add_traj


policy_list = [[traj_for(i)] for i in gs.player_list]
observer_list = [[traj_for(i)] for i in gs.player_list]
p = PyTanFast(policy_list, observer_list)


class TestHandlePlayMonopoly(unittest.TestCase):

	def test_general_use(self):
		p._reset()
		red, green, blue = p.player_list
		monopoly_index = 0
		red.resource_cards.fill(0)
		green.resource_cards += np.array([1, 1, 1, 1, 1], dtype=np.int8)
		blue.resource_cards += np.array([1, 1, 1, 1, 1], dtype=np.int8)
		p.handler.handle_play_monopoly(monopoly_index, red)
		self.assertEqual(red.resource_cards[monopoly_index], 2)
		self.assertEqual(np.sum(red.resource_cards[monopoly_index]), 2)

	def test_no_cards(self):
		p._reset()
		red, green, blue = p.player_list
		monopoly_index = 0
		green.resource_cards.fill(0)
		blue.resource_cards.fill(0)
		p.handler.handle_play_monopoly(monopoly_index, red)
		self.assertEqual(red.resource_cards[monopoly_index], 0)

	def test_plays_card(self):
		p._reset()
		red, green, blue = p.player_list
		monopoly_index = 0
		red.development_cards.fill(0)
		red.development_cards[gs.monopoly_index] += 1
		self.assertEqual(np.sum(red.development_cards), 1)
		p.handler.handle_play_monopoly(monopoly_index, red)
		self.assertEqual(np.sum(red.development_cards), 0)
