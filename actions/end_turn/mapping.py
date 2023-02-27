from actions.end_turn.handle import handle_end_turn
# from actions.end_turn.mask import mask_end_turn

end_turn = "END_TURN"


def get_end_turn_mapping():
	return {(end_turn, None): {"callback": (handle_end_turn, None),}}# "mask": mask_end_turn()}}
