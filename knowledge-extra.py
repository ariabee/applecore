# COPY of Knowledge class with future-needs exploring.
# Knowledge base class for game character.
# Contains background lexicon and movement functions.
# Contains mapping and updating functions of meaning 
# to pre-existing knowledge.

class Knowledge:

    def __init__(self, agent):
        # TODO: might need to make every word have a function. or a value of category, index in category
        self._lexicon = {'move': 0, 'run': 0, 'go': 0, 'walk': 0, \
                         'left': 1, \
						 'right': 2, \
                         'up': 3, \
                         'down': 4, \
                         'yes': 5, 'no': 6, \
						 'tree': 7} 

        self.actions = [self.move, self.left, self.right, self.up, self.down, self.yes, self.no] # Items might need to be lists of actions (such that lexemes map to action sequences)
        self.objects = {'tree':(10,10), \
                         'you': agent.position} # x, y posiiton on map
        self.confirmations = [self.yes, self.no]
        #self.parameters = []
        self.categories = {"action": self.actions, "object": self.objects, "confirm": self.confirmations}
        # self.categories = {"action": self.actions, "object": self.objects, "confirm": self.confirmations, \
        #                    "parameter": self.parameters}
                        
        
        self.agent = agent

    def lexicon(self):
        return self._lexicon

    def add_to_lexicon(self, words, action):
        self._lexicon.update({words : action})
        # might change action to self._actions[action]

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
        return("yes!")

    def no(self):
        return("oops :(")



    