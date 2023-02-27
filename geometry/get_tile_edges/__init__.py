def get_tile_edges(x, y):
	return [
		(x * 2, y * 2 + 1),
		(x * 2, y * 2 + 2),
		(x * 2 + 1, y),
		(x * 2 + 1, y + 1),
		(x * 2 + 2, y * 2),
		(x * 2 + 2, y * 2 + 1),
	]
