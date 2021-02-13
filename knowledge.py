# ~KNOWLEDGE v.0 : every lexeme maps to a function, no separate lexeme categories/types~
# Knowledge base class for game character.
# Contains background lexicon and movement functions.
# Contains mapping and updating functions of meaning 
# to pre-existing knowledge.

from transcript import Transcript

class Knowledge:

    def __init__(self, agent):
        # TODO: might need to make every word have a function. or a value of category, index in category
        self._lexicon = {'move': [0], 'run': [0], 'go': [0], 'walk': [0], \
                         'left': [1], \
						 'right': [2], \
                         'up': [3], \
                         'down': [4], \
                         'yes': [5], 'no': [6], \
						 'tree': [7], \
                         'you': [8], agent.name: [8] }
        
        self._learned = {} # An initially empty list of learned commands mapped to actions.

        self.actions = [self.move, self.left, self.right, self.up, self.down, self.yes, self.no, self.tree, self.me]
        self.objects = {'tree':(10,10), \
                         'me': agent.position} # x, y posiiton on map
        # self.confirmations = [self.yes, self.no]
        # self.categories = {"action": self.actions, "object": self.objects, "confirm": self.confirmations}

        self.agent = agent


    def lexicon(self):
        return self._lexicon

    def learned(self):
        return self._learned

    def add_to_lexicon(self, word, action):
        self._lexicon.update({word : action})
        # might change action to self._actions[action]
        print(self._lexicon)
    
    def add_to_learned(self, words, action_sequence):
        self._learned.update({words : action_sequence})
        print("~~learned: " + str(self._learned))
        return "I learned to: " + str(words)

    def link_prev_command(self):
        prior_input, prior_actions = self.agent.transcript.previous()
        response = self.add_to_learned(prior_input, prior_actions) 
        return response

    def is_transcript_empty(self):
        instruct_is_empty = not self.agent.transcript.instructions
        actions_is_empty = not self.agent.transcript.action_sequences
        return instruct_is_empty or actions_is_empty

    def move(self):
        return("moving")

    def left(self):
        return("going left")

    def right(self):
        return("going right")

    def up(self):
        return("going up")

    def down(self):
        return("going down")

    def yes(self):
        response = self.link_prev_command() if not self.is_transcript_empty() else ""
        return("yes! " + str(response))

    def no(self):
        return("oops :(")

    def tree(self):
        return self.objects['tree'] # Return tree coordinates

    def me(self):
        return self.objects['me'] # Return agent coordinates

    # def an_object(self, object_name):
    #     coordinates = self.objects[object_name]
    #     return coordinates

    # Define complex actions / game tasks:

    def climb_the_tree(self):
    # def climb_the_tree(self, position, action_sequence):
        # (should perform actions, either predefined or passed into the function)
        # move agent to position
        # for action in action_sequence:
        #     do action

        # IDEA: could pass in action squence parameter made up of agent position + actions taken by agent from 
        # last completed action sequence (marked success)
 
        return "climbing the tree"




    