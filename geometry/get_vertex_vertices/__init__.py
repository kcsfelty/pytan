def get_vertex_vertices(x, y):
	offset = (y % 2) * 2 - 1
	return [
		(x, y - 1),
		(x, y + 1),
		(x + offset, y - offset)
	]
