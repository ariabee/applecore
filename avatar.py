# Avatar class for game character. 
# Testing out python logic for mapping instructions to functions.


# Action functions
# def yes(instruct):
# 	# validate previous instruction mapped to movement

# 	# feedback
# 	return("Yay!")

# ACTIONS = [move(instruct), left(instruct), right(instruct), yes(instruct), no(instruct)]


# Game avatar
class Avatar:
	'''
	param: knowledge, a dictionary mapping of strings (commands) to enumerated actions (functions)
	'''
	def __init__(self):
		self.name = "Young Apple"
		self.knowledge = {'run': 0, 'move': 0, 'go': 0, 'left': 1, 'right': 2, 'yes': 3, 'no': 4}

	# def process(raw_instruction):
	# 	instruction = raw_instruction.split() # nvm bc want any string combo

	def give_name(self, new_name):
		self.name = new_name

	def try_action(self, instruction):

		composition = ""
		actions = []
		unknowns = ""

		for movement_words in self.knowledge:
			if movement_words in instruction:
				composition += (movement_words + " ")
				actions.append(self.knowledge[movement_words])

		return(composition, actions)





