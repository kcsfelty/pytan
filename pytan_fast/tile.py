import pytan_fast.settings as gs
import numpy as np

class Tile:
	def __repr__(self):
		return "Tile{}".format(self.index)

	def __init__(self, index, has_robber, resource, roll_number):
		self.index = index
		self.tiles = []
		self.vertices = []
		self.edges = []

		self.resource_index = None
		self.has_robber = has_robber
		self.resource = resource
		self.roll_number = roll_number
		self.players_to_rob = np.zeros((gs.player_count, gs.player_count), dtype=np.bool8)

	def reset(self):
		self.players_to_rob.fill(False)
