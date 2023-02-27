from actions.place_settlement.handle import handle_place_settlement
# from actions.place_settlement.mask import mask_place_settlement

place_settlement_prefix = "PLACE_SETTLEMENT"


def get_place_settlement_term(vertex):
	return place_settlement_prefix, vertex


def get_place_settlement_mapping(vertex_list):
	place_settlement_mapping = {}
	for vertex in vertex_list:
		term = get_place_settlement_term(vertex)
		callback = (handle_place_settlement, vertex)
		mask = None#mask_place_settlement(vertex)
		place_settlement_mapping[term] = {"callback": callback, "mask": mask}
	return place_settlement_mapping
