from game import game_settings
from geometry.get_edge_edges import get_edge_edges
from geometry.get_edge_list import get_edge_list
from geometry.get_edge_tiles import get_edge_tiles
from geometry.get_edge_vertices import get_edge_vertices
from geometry.get_tile_edges import get_tile_edges
from geometry.get_tile_list import get_tile_list
from geometry.get_tile_tiles import get_tile_tiles
from geometry.get_tile_vertices import get_tile_vertices
from geometry.get_vertex_edges import get_vertex_edges
from geometry.get_vertex_list import get_vertex_list
from geometry.get_vertex_tiles import get_vertex_tiles
from geometry.get_vertex_vertices import get_vertex_vertices


def get_geometry():
	tile_list = get_tile_list(game_settings.board_height, game_settings.board_width)
	vertex_list = get_vertex_list(tile_list)
	edge_list = get_edge_list(tile_list)

	tile_tiles = {}
	tile_vertices = {}
	tile_edges = {}
	for tile in tile_list:
		local_tiles = get_tile_tiles(*tile)
		tile_tiles[tile] = []
		for local_tile in local_tiles:
			if local_tile in tile_list:
				tile_tiles[tile].append(local_tile)
		local_vertices = get_tile_vertices(*tile)
		tile_vertices[tile] = []
		for local_vertex in local_vertices:
			if local_vertex in vertex_list:
				tile_vertices[tile].append(local_vertex)
		local_edges = get_tile_edges(*tile)
		tile_edges[tile] = []
		for local_edge in local_edges:
			if local_edge in edge_list:
				tile_edges[tile].append(local_edge)

	vertex_tiles = {}
	vertex_vertices = {}
	vertex_edges = {}
	for vertex in vertex_list:
		local_tiles = get_vertex_tiles(*vertex)
		vertex_tiles[vertex] = []
		for local_tile in local_tiles:
			if local_tile in tile_list:
				vertex_tiles[vertex].append(local_tile)
		local_vertices = get_vertex_vertices(*vertex)
		vertex_vertices[vertex] = []
		for local_vertex in local_vertices:
			if local_vertex in vertex_list:
				vertex_vertices[vertex].append(local_vertex)
		local_edges = get_vertex_edges(*vertex)
		vertex_edges[vertex] = []
		for local_edge in local_edges:
			if local_edge in edge_list:
				vertex_edges[vertex].append(local_edge)

	edge_tiles = {}
	edge_vertices = {}
	edge_edges = {}
	for edge in edge_list:
		local_tiles = get_edge_tiles(*edge)
		edge_tiles[edge] = []
		for local_tile in local_tiles:
			if local_tile in tile_list:
				edge_tiles[edge].append(local_tile)
		local_vertices = get_edge_vertices(*edge)
		edge_vertices[edge] = []
		for local_vertex in local_vertices:
			if local_vertex in vertex_list:
				edge_vertices[edge].append(local_vertex)
		local_edges = get_edge_edges(*edge)
		edge_edges[edge] = []
		for local_edge in local_edges:
			if local_edge in edge_list:
				edge_edges[edge].append(local_edge)

	return [
		(tile_list, tile_tiles, tile_vertices, tile_edges),
		(vertex_list, vertex_tiles, vertex_vertices, vertex_edges),
		(edge_list, edge_tiles, edge_vertices, edge_edges)
	]
