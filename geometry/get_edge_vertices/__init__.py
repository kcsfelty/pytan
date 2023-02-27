def get_edge_vertices(x, y):
	if x % 2 == 0:
		return [
			(x // 2, y),
			(x // 2, y + 1)
		]
	else:
		return [
			((x - 1) // 2, y * 2 + 1),
			((x - 1) // 2 + 1, y * 2)
		]
