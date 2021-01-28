# Testing out python logic for mapping instructions to functions
from avatar import *

# Define complex actions





def name_apple():
	'''
	User names the game avatar. Returns string name of avatar. 
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

# Create avatar
avatar = Avatar()

# Give Young Apple a name
name = name_apple()
avatar.give_name(name)


# Game Loop
print("Teach " + name + " to: climb the tree.")
instruction = ""
while instruction != "stop":

	instruction = input("\nType something: ").lower()

	trying = avatar.try_action(instruction)
	print(name + ": ", end="")
	print(trying)
	
	


	




