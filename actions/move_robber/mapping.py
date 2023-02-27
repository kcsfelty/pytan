from actions.move_robber.handle import handle_move_robber
# from actions.move_robber.mask import mask_move_robber

move_robber_prefix = "MOVE_ROBBER"


def get_move_robber_term(tile):
	return move_robber_prefix, tile


def get_move_robber_mapping(tile_list):
	move_robber_mapping = {}
	for tile in tile_list:
		term = get_move_robber_term(tile)
		callback = (handle_move_robber, tile)
		mask = None #mask_move_robber(tile)
		move_robber_mapping[term] = {"callback": callback, "mask": mask}
	return move_robber_mapping
