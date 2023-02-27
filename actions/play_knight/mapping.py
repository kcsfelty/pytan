from actions.play_knight.handle import handle_play_knight
# from actions.play_knight.mask import mask_play_knight

play_knight = "PLAY_KNIGHT"


def get_play_knight_mapping():
	return {(play_knight, None): {"callback": (handle_play_knight, None),}}# "mask": mask_play_knight()}}
