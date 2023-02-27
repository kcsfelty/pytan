from actions.place_road.handle import handle_place_road
# from actions.place_road.mask import mask_place_road

place_road_prefix = "PLACE_ROAD"


def get_place_road_term(edge):
	return place_road_prefix, edge


def get_place_road_mapping(edge_list):
	place_road_mapping = {}
	for edge in edge_list:
		term = get_place_road_term(edge)
		callback = (handle_place_road, edge)
		mask = None #mask_place_road(edge)
		place_road_mapping[term] = {"callback": callback, "mask": mask}
	return place_road_mapping
