import random

default_dice_count = 2
default_dice_min = 0
default_dice_max = 5


class Dice:
	def __init__(self, dice_count=default_dice_count, dice_min=default_dice_min, dice_max=default_dice_max, seed=None):
		if seed: random.seed(seed)
		self.dice_count = dice_count
		self.dice_min = dice_min
		self.dice_max = dice_max
		self.max_roll = self.dice_max * self.dice_count
		self.min_roll = self.dice_min * self.dice_count
		self.dice_range = self.max_roll - self.min_roll
		self.last_roll = -1
		self.histogram = [0 for _ in range((self.dice_range + 1))]

	def reset(self):
		self.last_roll = -1
		self.histogram = [0 for _ in range((self.dice_range + 1))]

	def roll(self):
		total = 0
		for _ in range(self.dice_count):
			total += random.randint(self.dice_min, self.dice_max)
		self.last_roll = total
		self.histogram[total] += 1
		return total
