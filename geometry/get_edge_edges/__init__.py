from geometry.get_edge_vertices import get_edge_vertices
from geometry.get_vertex_edges import get_vertex_edges


def get_edge_edges(x, y):
	edges = []
	for x, y in get_edge_vertices(x, y):
		for edge in get_vertex_edges(x, y):
			if edge not in edges:
				edges.append(edge)
	return edges
