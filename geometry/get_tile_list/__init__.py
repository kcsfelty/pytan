from geometry.get_imaginary_tiles import get_imaginary_tiles


def get_tile_list(h=5, w=5, prune_imaginary=True):
	tile_list = []
	imaginary_tiles = get_imaginary_tiles(h, w)
	for x in range(w):
		for y in range(h):
			if (x, y) not in imaginary_tiles and prune_imaginary:
				tile_list.append((x, y))
	return tile_list
