def get_tile_vertices(x, y):
	return [

		(x, 2 * y + 2),
		(x, 2 * y + 3),
		(x + 1, 2 * y + 2),
		(x + 1, 2 * y + 1),
		(x + 1, 2 * y + 0),
		(x, 2 * y + 1),
	]
