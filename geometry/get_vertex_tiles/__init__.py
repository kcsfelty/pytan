def get_vertex_tiles(x, y):
	odd_vertex = y % 2
	axis = (y - odd_vertex) // 2
	return [
		(x - 1, axis),
		(x, axis - 1),
		(x - 1 + odd_vertex, axis - 1 + odd_vertex)
	]
