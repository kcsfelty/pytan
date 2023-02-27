from geometry.get_tile_vertices import get_tile_vertices


def get_vertex_list(tile_list):
	vertex_list = []
	for x, y in tile_list:
		for vertex in get_tile_vertices(x, y):
			if vertex not in vertex_list:
				vertex_list.append(vertex)
	return vertex_list
