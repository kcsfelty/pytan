from actions.place_city.handle import handle_place_city
# from actions.place_city.mask import mask_place_city

place_city_prefix = "PLACE_CITY"


def get_place_city_term(vertex):
	return place_city_prefix, vertex


def get_place_city_mapping(vertex_list):
	place_city_mapping = {}
	for vertex in vertex_list:
		term = get_place_city_term(vertex)
		callback = (handle_place_city, vertex)
		mask = None# mask_place_city(vertex)
		place_city_mapping[term] = {"callback": callback, "mask": mask}
	return place_city_mapping
