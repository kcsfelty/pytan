import reference.settings as gs
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
from board.edge import Edge
from board.tile import Tile
from board.vertex import Vertex
from reference import settings
from reference.settings import desert_index
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

		self.roll_hash = [[[] for _ in gs.roll_list] for _ in range(self.game.game_count)]

		self.robbed_tile = [None for _ in range(self.game.game_count)]

		self.port_vertex_lookup = [(self.vertex_hash[port_a], self.vertex_hash[port_b]) for port_a, port_b in settings.port_vertex_groups]

		self.tiles = [Tile(
			key=tile,
			index=self.tile_hash[tile],
			batch_size=self.game.batch_size,
			has_robber=self.state.tile_has_robber[:, self.tile_hash[tile]:self.tile_hash[tile]+1],
			resource=self.state.tile_resource[:, self.tile_hash[tile], :],
			roll_number=self.state.tile_roll_number[:, self.tile_hash[tile], :],
		) for tile in self.tile_list]

		self.vertices = [Vertex(
			key=vertex,
			index=self.vertex_hash[vertex],
			game_count=self.game.game_count,
			settlement=self.state.vertex_settlement[:, self.vertex_hash[vertex]:self.vertex_hash[vertex]+1],
			city=self.state.vertex_city[:, self.vertex_hash[vertex]:self.vertex_hash[vertex]+1],
			is_open=self.state.vertex_open[:, self.vertex_hash[vertex]:self.vertex_hash[vertex]+1],
			port=self.state.vertex_has_port[:, self.vertex_hash[vertex]]
		) for vertex in self.vertex_list]

		self.edges = [Edge(
			key=edge,
			index=self.edge_hash[edge],
			is_open=self.state.edge_open[:, self.edge_hash[edge]:self.edge_hash[edge]+1],
		) for edge in self.edge_list]

		# Connect adjacent geometries together
		for tile in self.tiles:
			for adjacent_tile in get_tile_tiles(*tile.key):
				if adjacent_tile in self.tile_list:
					tile.tiles.append(self.tiles[self.tile_hash[adjacent_tile]])
			for adjacent_vertex in get_tile_vertices(*tile.key):
				if adjacent_vertex in self.vertex_list:
					tile.vertices.append(self.vertices[self.vertex_hash[adjacent_vertex]])
			for adjacent_edge in get_tile_edges(*tile.key):
				if adjacent_edge in self.edge_list:
					tile.edges.append(self.edges[self.edge_hash[adjacent_edge]])
		for vertex in self.vertices:
			for adjacent_tile in get_vertex_tiles(*vertex.key):
				if adjacent_tile in self.tile_list:
					vertex.tiles.append(self.tiles[self.tile_hash[adjacent_tile]])
			for adjacent_vertex in get_vertex_vertices(*vertex.key):
				if adjacent_vertex in self.vertex_list:
					vertex.vertices.append(self.vertices[self.vertex_hash[adjacent_vertex]])
			for adjacent_edge in get_vertex_edges(*vertex.key):
				if adjacent_edge in self.edge_list:
					vertex.edges.append(self.edges[self.edge_hash[adjacent_edge]])
		for edge in self.edges:
			for adjacent_tile in get_edge_tiles(*edge.key):
				if adjacent_tile in self.tile_list:
					edge.tiles.append(self.tiles[self.tile_hash[adjacent_tile]])
			for adjacent_vertex in get_edge_vertices(*edge.key):
				if adjacent_vertex in self.vertex_list:
					edge.vertices.append(self.vertices[self.vertex_hash[adjacent_vertex]])
			for adjacent_edge in get_edge_edges(*edge.key):
				if adjacent_edge in self.edge_list:
					edge.edges.append(self.edges[self.edge_hash[adjacent_edge]])

		for vertex in self.vertices:
			vertex.build_adjacent_vertex_edge()
		for edge in self.edges:
			edge.build_adjacent_edge_vertex()

	def reset(self, game_index, board_layout=None, port_layout=None):
		self.roll_hash[game_index] = [[] for _ in gs.roll_list]
		self.robbed_tile[game_index] = None
		tile_resource_layout, tile_roll_number_layout = board_layout or self.get_board_layout()
		for tile, resource_index, roll_index in zip(self.tiles, tile_resource_layout, tile_roll_number_layout):
			if roll_index != -1:
				tile.resource[game_index][resource_index] = 1
				tile.roll_number[game_index][roll_index] = 1
				self.roll_hash[game_index][roll_index].append(tile)
			else:
				tile.has_robber[game_index].fill(1)
				self.robbed_tile[game_index] = tile
		port_layout = port_layout or self.get_port_layout()
		for vertices, port_type in zip(self.port_vertex_lookup, port_layout):
			vertex_a, vertex_b = vertices
			self.vertices[vertex_a].port[game_index][port_type] = 1
			self.vertices[vertex_b].port[game_index][port_type] = 1

	def get_board_layout(self):
		tile_resource_count = reverse_histogram([x for x in settings.tile_resource_count])
		tile_roll_number_count = reverse_histogram([x for x in settings.tile_roll_number_count_per_type])
		tile_roll_number_count.insert(tile_resource_count.index(desert_index), -1)
		return tile_resource_count, tile_roll_number_count

	def get_port_layout(self):
		port_type_count = reverse_histogram([x for x in settings.port_type_count_per_type])
		return port_type_count
