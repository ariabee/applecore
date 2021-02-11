# Knowledge base class for game character.
# Contains background lexicon and movement functions.
# Contains mapping and updating functions of meaning 
# to pre-existing knowledge.

class Knowledge:

    def __init__(self, agent):
        self.lexicon = {'move': 0, 'run': 0, 'go': 0, 'walk': 0, 'left': 1, \
						  'right': 2, 'yes': 3, 'no': 4, \
						  'tree': 5}

        self.actions = [self.move, self.left, self.right, self.up, self.down, self.yes, self.no]
        self.objects = {'tree':(10,10), \
                         'you': agent.position} # x, y posiiton on map
        self.categories = [self.actions, self.objects]

        self.agent = agent

    def move(self):
        return("moving")

    def left(self):
        return("going left")

    def right(self):
        return("going right")

    def up(self):
        return("going up")

    def down(self):
        return("going up")

    def yes():
        return("yes!")

    def no(self):
        return("oops :(")



    