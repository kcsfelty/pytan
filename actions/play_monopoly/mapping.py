from actions.play_monopoly.handle import handle_play_monopoly
# from actions.play_monopoly.mask import mask_play_monopoly

play_monopoly_prefix = "PLAY_MONOPOLY"


def get_play_monopoly_term(resource_index):
	return play_monopoly_prefix, resource_index


def get_play_monopoly_mapping(resource_list):
	play_monopoly_mapping = {}
	for resource_index in resource_list:
		term = get_play_monopoly_term(resource_index)
		callback = (handle_play_monopoly, resource_index)
		mask = None#mask_play_monopoly(resource_index)
		play_monopoly_mapping[term] = {"callback": callback, "mask": mask}
	return play_monopoly_mapping
