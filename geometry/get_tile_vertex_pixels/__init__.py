def get_tile_vertex_pixels(x, y, r):
	return [
		(x - r, y + r / 2),
		(x, y + r),
		(x + r, y + r / 2),
		(x + r, y - r / 2),
		(x, y - r),
		(x - r, y - r / 2),
	]