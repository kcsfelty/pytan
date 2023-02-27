from geometry.get_vertex_edges import get_vertex_edges


def get_vertices_edge(vertex_a, vertex_b):
	edge_list_a = get_vertex_edges(*vertex_a)
	edge_list_b = get_vertex_edges(*vertex_b)
	for edge_a in edge_list_a:
		for edge_b in edge_list_b:
			if edge_a == edge_b:
				return edge_a
