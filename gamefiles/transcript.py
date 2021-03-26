# Transcript class for voice game.
# Class for recording all input/instructions from the user,
# all corresponding action sequences carried out by agent,
# and all linguistic feedback given by the agent
# for a single game.
#
# Transcript will be accessed by agent to recall previous "experience".
# TODO: after baseline, expand and refine linguistic feedback property
from datetime import datetime
import csv

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
        self.timestamp = self.__init_timestamp()
        self.file = 'transcript_' + self.timestamp + '.csv'
        self.file_path = 'transcripts/' + self.file

    
    def __init_timestamp(self):
        '''
        Creates the current timestamp for the transcript file name.
        Returns a string in the form dd_mm_yyyy_HH_MM.
        '''
        today = datetime.today()
        date = str(today.day)+'_'+str(today.month)+'_'+str(today.year)
        time = datetime.now().strftime("%H_%M")
        return date + '_' + time

    
    def store_instruction(self, instruct):
        self.instructions.append(instruct)


    def store_actions(self, actions): 
        self.action_sequences.append(actions)


    def store(self, instruct="", actions=[]):
        """
        Store instructions: string, and actions: list in transcript.
        """
        self.instructions.append(instruct)
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
        index_prev_input = len(self.instructions)-1-1 # assume that all instructions are stored first
        return self.action_sequences[index_prev_input]


    def current_actions(self):
        '''
        Returns the current action sequence from the transcript.
        '''
        index_current_input = len(self.instructions)-1
        return self.action_sequences[index_current_input]


    def previous(self):
        '''
        Returns the previous instruction and action sequence in the transcript as a tuple.
        '''
        return (self.previous_instruction(), self.previous_actions())


    def current(self):
        '''
        Returns the most recent instruction and action sequence in the transcript as a tuple.
        '''
        return (self.current_instruction(), self.current_actions())


    def is_empty(self):
        instruct_is_empty = not self.instructions
        actions_is_empty = not self.action_sequences
        return instruct_is_empty or actions_is_empty

    def entry_number(self):
        entry_number = len(self.instructions) - 1
        return entry_number

    def save(self):
        '''
        Saves transcript to a csv file.
        '''
        # TODO: update to also save responses/feedback
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Instruction', 'Actions', 'Response'])

            if self.instructions and self.action_sequences:
                for instruct, action in zip(self.instructions, self.action_sequences):
                    writer.writerow([str(instruct), str(action), ''])
                    