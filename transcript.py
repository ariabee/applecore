# Transcript class for voice game.
# Class for recording all input/instructions from the user,
# all corresponding action sequences carried out by agent,
# and all linguistic feedback given by the agent
# for a single game.
#
# Transcript will be accessed by agent to recall previous "experience".
# TODO: think about time stamps / time stamp approaches in future dev. iterations.
# TODO: after baseline, expand and refine linguistic feedback property

class Transcript:

    def __init__(self):
        '''
        Create a new transcript for the game.

        param: instructions, list of string instructions from beginning (0) to end (length - 1)
        param: action_sequences, list of list action sequences
        param: feedback, list of string feedback OR list of tuples(string response, string facial expression)
        '''
        self.instructions = []
        self.action_sequences = []
        self.feedback = [] # Will be added after baseline.


    def store_instruction(self, instruct):
        self.instructions.append(instruct)


    def store_actions(self, actions): 
        self.action_sequences.append(actions)


    def store(self, instruct="", actions=[]):
        self.instructions.append(instruct)
        if actions:
            self.action_sequences.append(actions)


    def previous_instruction(self):
        '''
        Returns the previous instruction from the transcript.
        '''
        next_to_last_index = len(self.instructions)-1-1
        return self.instructions[next_to_last_index]


    def current_instruction(self):
        '''
        Returns the current instruction from the transcript.
        '''
        return self.instructions[len(self.instructions)-1]
   

    def previous_actions(self):
        '''
        Returns the previous action sequence from the transcript.
        '''
        return self.action_sequences[len(self.action_sequences)-1-1]


    def current_actions(self):
        '''
        Returns the current action sequence from the transcript.
        '''
        return self.action_sequences[len(self.action_sequences)-1]


    def current(self):
        '''
        Returns the most recent instruction and action sequence in the transcript as a tuple.
        '''
        return (self.current_instruction(), self.current_actions())