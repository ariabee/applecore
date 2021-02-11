# Testing out python logic for mapping instructions to functions
from agent import *

# Define complex actions





def name_agent():
	'''
	User names the game agent. Returns string name of agent. 
	'''
	name = input("What would you like to call Young Apple when teaching it tricks? ")
	confirm = input("Call Young Apple, '" + name + "'? (y/n) ")
	while confirm.lower()=="n":
		name = input("Okay, what would you like to call Young Apple? ")
		confirm = input("Call Young Apple, '" + name + "'? (y/n) ")
	print("Terrific. '" + name +"' it is." )

	return(name)


# Introduction
print("\n        *******************************************************\n\
	* Hello, and welcome to the world of me, Young Apple. *\n\
	* I'm ready to move around and learn new tricks.      *\n\
	*******************************************************\n")

# Create agent
agent = Agent()

# Give Young Apple a name
name = name_agent()
agent.give_name(name)


# Game Loop
print("Teach " + agent.name + " to: climb the tree.")
instruction = ""
while instruction != "stop":

	# Get text instruction from user
	instruction = input("\nType something: ").lower()

	# Process instruction, output text interpretation and action
	attempt = agent.attempt(instruction)
	print(agent.name + ": ", end="")
	print(attempt)
	
	


	





