from geometry.get_tile_edges import get_tile_edges


def get_edge_list(tile_list):
	edge_list = []
	for x, y in tile_list:
		for edge in get_tile_edges(x, y):
			if edge not in edge_list:
				edge_list.append(edge)
	return edge_list

