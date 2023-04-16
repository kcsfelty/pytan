class Vertex:
	def __repr__(self):
		return "Vertex{}".format(self.key)

	def __init__(self, index, key, game_count, settlement, city, is_open, port):
		self.index = index
		self.key = key
		self.settlement = settlement
		self.city = city
		self.open = is_open
		self.port = port
		self.owned_by = [None for _ in range(game_count)]
		self.tiles = []
		self.vertices = []
		self.edges = []
		self.edge_between_vertex = {}

	def reset(self, game_index):
		self.owned_by[game_index] = None

	def build_adjacent_vertex_edge(self):
		for adjacent_vertex in self.vertices:
			for local_edge in self.edges:
				for adjacent_edge in adjacent_vertex.edges:
					if local_edge.index == adjacent_edge.index:
						self.edge_between_vertex[adjacent_vertex] = local_edge
