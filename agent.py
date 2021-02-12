# Agent class for game character. 
# Testing out python logic for mapping instructions to functions.

from knowledge import Knowledge


# Action functions
# def yes(instruct):
# 	# validate previous instruction mapped to movement

# 	# feedback
# 	return("Yay!")

# ACTIONS = [move(instruct), left(instruct), right(instruct), yes(instruct), no(instruct)]

def move(instruct):

	response = instruct + ": *agent is moving*"
	return(response)



# Game agent
class Agent:
	'''
	param: knowledge, a dictionary mapping of strings (commands) to enumerated actions (functions)
	'''
	def __init__(self):
		self.name = "Young Apple"
		self.position = (20,20) # testing position
		self.knowledge = Knowledge(self)
		# self.knowledge = {'move': 0, 'run': 0, 'go': 0, 'walk': 0, 'left': 1, \
		# 				  'right': 2, 'yes': 3, 'no': 4, \
		# 				  'tree': 5}


	# def process(raw_instruction):
	# 	instruction = raw_instruction.split() # nvm bc want any string combo

	def give_name(self, new_name):
		self.name = new_name


	def interpret(self, instruction):
		"""
		The Agent processes the instruction into 
		1) words "it understands" / that are retrievable in the knowledge base
		2) a list of actions to carry out

		param: instruction, the input string from the user

		return: composition, the recognized string of words
		return: actions, the list of corresponding actions 
		"""

		composition = ""
		actions = []
		unknowns = ""

		instruction = instruction.split() # split sentence into list of words
		lexicon =  self.knowledge.lexicon()

		# for words in lexicon:
		# 	if words in instruction: # make sure this is in order
		# Check for movement words in the instruction that the agent also recognizes
		for words in instruction:
			if words in lexicon:
				composition += (words + " ")
				actions.append(lexicon[words])

		#return(composition, actions)
		return(composition, self.try_actions(actions))


	def try_actions(self, parsed_actions):
		"""
		Execute the retrieved action functions.
		"""
		responses = []

		for action in parsed_actions:
			action_response = self.knowledge.actions[action]() # do the action, get the response
			responses.append(action_response) # todo: update responses to be specific to user's words 
											  # (through a function parameter or through function itself)

		return(responses)

		












