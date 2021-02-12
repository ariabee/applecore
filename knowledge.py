# Knowledge base class for game character.
# Contains background lexicon and movement functions.
# Contains mapping and updating functions of meaning 
# to pre-existing knowledge.

class Knowledge:

    def __init__(self, agent):
        self._lexicon = {'move': 0, 'run': 0, 'go': 0, 'walk': 0, \
                         'left': 1, \
						 'right': 2, \
                         'up': 3, \
                         'down': 4, \
                         'yes': 5, 'no': 6, \
						 'tree': 7} # TODO: might need to make every word have a function. or a value of category, index in category

        self.actions = [self.move, self.left, self.right, self.up, self.down, self.yes, self.no]
        self.objects = {'tree':(10,10), \
                         'you': agent.position} # x, y posiiton on map
        self.confirmations = [self.yes, self.no]
        self.categories = [self.actions, self.objects, self.confirmations,]
        
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



    