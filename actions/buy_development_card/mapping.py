from actions.buy_development_card.handle import handle_buy_development_card
# from actions.buy_development_card.mask import mask_buy_development_card

buy_development_card = "BUY_DEVELOPMENT_CARD"


def get_buy_development_card_mapping():
	return {(buy_development_card, None): {"callback": (handle_buy_development_card, None),}}   #"mask": mask_buy_development_card()}}
