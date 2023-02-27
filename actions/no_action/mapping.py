from actions.no_action.handle import handle_no_action
# from actions.no_action.mask import mask_no_action

no_action = "NO_ACTION"


def get_no_action_mapping():
	return {(no_action, None): {"callback": (handle_no_action, None),}}# "mask": mask_no_action()}}
