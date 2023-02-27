import pytan_fast.definitions as df


class Edge:
	def __repr__(self):
		return "Edge{}".format(self.index)

	def __init__(self, index, is_open):
		self.index = index
		self.open = is_open
		self.tiles = []
		self.vertices = []
		self.edges = []
		self.adjacent_edge_vertex = {}

	def reset(self):
		pass

	def build_adjacent_edge_vertex(self):
		for adjacent_edge in self.edges:
			for local_vertex in self.vertices:
				for adjacent_vertex in adjacent_edge.vertices:
					if local_vertex.index == adjacent_vertex.index:
						self.adjacent_edge_vertex[adjacent_edge] = local_vertex
