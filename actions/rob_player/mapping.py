from actions.rob_player.handle import handle_rob_player
# from actions.rob_player.mask import mask_rob_player
from pytan_fast import settings

rob_player_prefix = "ROB_PLAYER"


def get_rob_player_term(rob_player_index):
	return rob_player_prefix, rob_player_index


def get_rob_player_mapping(player_list=settings.player_list):
	rob_player_mapping = {}
	for rob_player_index in player_list:
		term = get_rob_player_term(rob_player_index)
		callback = (handle_rob_player, rob_player_index)
		mask = None#mask_rob_player(rob_player_index)
		rob_player_mapping[term] = {"callback": callback, "mask": mask}
	return rob_player_mapping
