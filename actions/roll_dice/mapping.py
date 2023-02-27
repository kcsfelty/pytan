from actions.roll_dice.handle import handle_roll_dice
# from actions.roll_dice.mask import mask_roll_dice

roll_dice = "ROLL_DICE"


def get_roll_dice_mapping():
	return {(roll_dice, None): {"callback": (handle_roll_dice, None),}}# "mask": mask_roll_dice()}}
