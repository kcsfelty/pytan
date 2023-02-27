def get_vertex_edges(x, y):
	axis = y - y % 2
	offset = y % 2
	return [
		(x * 2, axis),
		(x * 2, axis + offset * 2 - 1),
		(x * 2 + offset * 2 - 1, axis // 2)
	]
