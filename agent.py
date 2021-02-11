# Agent class for game character. 
# Testing out python logic for mapping instructions to functions.


# Action functions
# def yes(instruct):
# 	# validate previous instruction mapped to movement

# 	# feedback
# 	return("Yay!")

# ACTIONS = [move(instruct), left(instruct), right(instruct), yes(instruct), no(instruct)]

def move(instruct):

	response = instruct + ": *agent is moving*"
	return(response)


ACTIONS = [move]


# Game agent
class Agent:
	'''
	param: knowledge, a dictionary mapping of strings (commands) to enumerated actions (functions)
	'''
	def __init__(self):
		self.name = "Young Apple"
		self.position = (20,20) # testing position
		self.knowledge = {'move': 0, 'run': 0, 'go': 0, 'walk': 0, 'left': 1, \
						  'right': 2, 'yes': 3, 'no': 4, \
						  'tree': 5}

	# def process(raw_instruction):
	# 	instruction = raw_instruction.split() # nvm bc want any string combo

	def give_name(self, new_name):
		self.name = new_name


	def attempt(self, instruction):
		"""
		The Agent processes the instruction into 
		1) words "it understands" / that are retrievable in the knowledge base.
		2) carries out the retrieved actions

		param: instruction, the input string from the user
		"""

		composition = ""
		actions = []
		unknowns = ""

		for movement_words in self.knowledge:
			if movement_words in instruction:
				composition += (movement_words + " ")
				actions.append(self.knowledge[movement_words])

		return(composition, actions)
		#return(composition, self.try_actions(actions))


	def try_actions(self, actions):
		"""
		Execute the retrieved action functions.
		"""
		responses = []

		for action_index in actions:
			responses.append(ACTIONS[action_index]("move")) # Update the parameter to be specific to user's words, or update move function

		return(responses)

		












