class Edge:
	def __repr__(self):
		return "Edge{}".format(self.key)

	def __init__(self, index, key, is_open):
		self.index = index
		self.key = key
		self.open = is_open
		self.tiles = []
		self.vertices = []
		self.edges = []
		self.adjacent_edge_vertex = {}

	def reset(self, game_index):
		pass

	def build_adjacent_edge_vertex(self):
		for adjacent_edge in self.edges:
			for local_vertex in self.vertices:
				for adjacent_vertex in adjacent_edge.vertices:
					if local_vertex.index == adjacent_vertex.index:
						self.adjacent_edge_vertex[adjacent_edge] = local_vertex
