def get_imaginary_tiles(h, w):
	if h < w:
		return False
	imaginary_tiles = []
	for x in range(w):
		for y in range(h):
			if x + y < (h // 2) or w + w // 2 <= x + y:
				imaginary_tiles.append((x, y))
	return imaginary_tiles