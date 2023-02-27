from geometry.get_edge_vertices import get_edge_vertices
from geometry.get_tile_edges import get_tile_edges
from geometry.get_tile_list import get_tile_list
from geometry.get_tile_vertex_pixels import get_tile_vertex_pixels
from geometry.get_tile_vertices import get_tile_vertices


def get_geometry_pixels(r=1):
	tile_list = get_tile_list()

	tile_pixels = {}
	vertex_pixels = {}
	edge_pixels = {}

	for tile in tile_list:
		t_x, t_y = tile
		t_p = (t_x * 2 + r * t_y, 1 * (t_y * 2 - (r / 2) * t_y))
		if tile not in tile_pixels:
			tile_pixels[tile] = t_p
		tile_vertices_pixels = get_tile_vertex_pixels(t_x, t_y, r)
		tile_vertices = get_tile_vertices(*tile)
		for vertex_location, vertex in zip(tile_vertices_pixels, tile_vertices):
			v_x, v_y = vertex_location
			v_p = (t_x + v_x + r * t_y, 1 * (t_y + v_y - (r / 2) * t_y))
			if vertex not in vertex_pixels:
				vertex_pixels[vertex] = v_p
	for tile in tile_list:
		for edge in get_tile_edges(*tile):
			edge_pixels[edge] = get_edge_vertices(*edge)
	return tile_pixels, vertex_pixels, edge_pixels
