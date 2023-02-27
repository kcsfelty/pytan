import pytan_fast.settings as gs
from geometry.get_edge_edges import get_edge_edges
from geometry.get_edge_list import get_edge_list
from geometry.get_edge_tiles import get_edge_tiles
from geometry.get_edge_vertices import get_edge_vertices
from geometry.get_tile_edges import get_tile_edges
from geometry.get_tile_list import get_tile_list
from geometry.get_tile_tiles import get_tile_tiles
from geometry.get_tile_vertices import get_tile_vertices
from geometry.get_vertex_edges import get_vertex_edges
from geometry.get_vertex_list import get_vertex_list
from geometry.get_vertex_tiles import get_vertex_tiles
from geometry.get_vertex_vertices import get_vertex_vertices
from pytan_fast import settings
from pytan_fast.definitions import tile_has_robber, tile_resource, tile_roll_number, vertex_settlement, vertex_city, \
	vertex_open, edge_open, vertex_has_port
from pytan_fast.edge import Edge
from pytan_fast.settings import desert_index
from pytan_fast.tile import Tile
from pytan_fast.vertex import Vertex
from util.reverse_histogram import reverse_histogram


class Board:
	def __init__(self, state, game):
		self.state = state
		self.game = game
		self.tile_list = get_tile_list()
		self.vertex_list = get_vertex_list(self.tile_list)
		self.edge_list = get_edge_list(self.tile_list)
		self.tile_hash = {tile: i for tile, i in zip(self.tile_list, range(len(self.tile_list)))}
		self.vertex_hash = {vertex: i for vertex, i in zip(self.vertex_list, range(len(self.vertex_list)))}
		self.edge_hash = {edge: i for edge, i in zip(self.edge_list, range(len(self.edge_list)))}

		self.roll_hash = [[] for _ in gs.roll_list]

		self.robbed_tile = None

		self.tiles = [Tile(
			index=tile,
			has_robber=self.state.game_state_slices[tile_has_robber][self.tile_hash[tile]:self.tile_hash[tile]+1],
			resource=self.state.game_state_slices[tile_resource][self.tile_hash[tile], :],
			roll_number=self.state.game_state_slices[tile_roll_number][self.tile_hash[tile], :],
		) for tile in self.tile_list]

		self.vertices = [Vertex(
			index=vertex,
			settlement=self.state.game_state_slices[vertex_settlement][self.vertex_hash[vertex]:self.vertex_hash[vertex]+1],
			city=self.state.game_state_slices[vertex_city][self.vertex_hash[vertex]:self.vertex_hash[vertex]+1],
			is_open=self.state.game_state_slices[vertex_open][self.vertex_hash[vertex]:self.vertex_hash[vertex]+1],
			port=self.state.game_state_slices[vertex_has_port][self.vertex_hash[vertex]]
		) for vertex in self.vertex_list]

		self.edges = [Edge(
			index=edge,
			is_open=self.state.game_state_slices[edge_open][self.edge_hash[edge]:self.edge_hash[edge]+1],
		) for edge in self.edge_list]

		# Connect adjacent geometries together
		for tile in self.tiles:
			for adjacent_tile in get_tile_tiles(*tile.index):
				if adjacent_tile in self.tile_list:
					tile.tiles.append(self.tiles[self.tile_hash[adjacent_tile]])
			for adjacent_vertex in get_tile_vertices(*tile.index):
				if adjacent_vertex in self.vertex_list:
					tile.vertices.append(self.vertices[self.vertex_hash[adjacent_vertex]])
			for adjacent_edge in get_tile_edges(*tile.index):
				if adjacent_edge in self.edge_list:
					tile.edges.append(self.edges[self.edge_hash[adjacent_edge]])
		for vertex in self.vertices:
			for adjacent_tile in get_vertex_tiles(*vertex.index):
				if adjacent_tile in self.tile_list:
					vertex.tiles.append(self.tiles[self.tile_hash[adjacent_tile]])
			for adjacent_vertex in get_vertex_vertices(*vertex.index):
				if adjacent_vertex in self.vertex_list:
					vertex.vertices.append(self.vertices[self.vertex_hash[adjacent_vertex]])
			for adjacent_edge in get_vertex_edges(*vertex.index):
				if adjacent_edge in self.edge_list:
					vertex.edges.append(self.edges[self.edge_hash[adjacent_edge]])
		for edge in self.edges:
			for adjacent_tile in get_edge_tiles(*edge.index):
				if adjacent_tile in self.tile_list:
					edge.tiles.append(self.tiles[self.tile_hash[adjacent_tile]])
			for adjacent_vertex in get_edge_vertices(*edge.index):
				if adjacent_vertex in self.vertex_list:
					edge.vertices.append(self.vertices[self.vertex_hash[adjacent_vertex]])
			for adjacent_edge in get_edge_edges(*edge.index):
				if adjacent_edge in self.edge_list:
					edge.edges.append(self.edges[self.edge_hash[adjacent_edge]])

		for vertex in self.vertices:
			vertex.build_adjacent_vertex_edge()
		for edge in self.edges:
			edge.build_adjacent_edge_vertex()

	def reset(self):
		tile_resource_count = reverse_histogram([x for x in settings.tile_resource_count])
		tile_roll_number_count = reverse_histogram([x for x in settings.tile_roll_number_count_per_type])
		tile_roll_number_count.insert(tile_resource_count.index(desert_index), -1)
		self.roll_hash = [[] for _ in gs.roll_list]
		for tile, resource_index, roll_index in zip(self.tiles, tile_resource_count, tile_roll_number_count):
			if roll_index != -1:
				tile.resource[resource_index].fill(1)
				tile.roll_number[roll_index].fill(1)
				tile.resource_index = resource_index
				self.roll_hash[roll_index].append(tile)
			else:
				tile.has_robber.fill(1)
				self.robbed_tile = tile

		port_type_count = reverse_histogram([x for x in settings.port_type_count_per_type])
		port_vertex_groups = [(self.vertex_hash[port_a], self.vertex_hash[port_b]) for port_a, port_b in settings.port_vertex_groups]
		for vertices, port_type in zip(port_vertex_groups, port_type_count):
			vertex_a, vertex_b = vertices
			self.vertices[vertex_a].port[port_type].fill(1)
			self.vertices[vertex_b].port[port_type].fill(1)
