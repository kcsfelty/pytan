import numpy as np
import pytan_fast.definitions as df
from pytan_fast.settings import tile_roll_number_type_count, resource_type_count_tile


class Vertex:
	def __repr__(self):
		return "Vertex{}".format(self.key)

	def __init__(self, index, key, settlement, city, is_open, port):
		self.index = index
		self.key = key
		self.settlement = settlement
		self.city = city
		self.open = is_open
		self.port = port
		self.owned_by = None
		self.tiles = []
		self.vertices = []
		self.edges = []
		self.edge_between_vertex = {}

	def reset(self):
		self.owned_by = None

	def build_adjacent_vertex_edge(self):
		for adjacent_vertex in self.vertices:
			for local_edge in self.edges:
				for adjacent_edge in adjacent_vertex.edges:
					if local_edge.index == adjacent_edge.index:
						self.edge_between_vertex[adjacent_vertex] = local_edge
