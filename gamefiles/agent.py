import pygame as pg
from settings import *
from map import collide_hit_rect
from sprites import *
import speech_recognition as sr
from asr.speech_to_text import SpeechToText
from knowledge import Knowledge
from transcript import Transcript
import math
from os import path
import time
import random

vec = pg.math.Vector2
vec_dest = pg.math.Vector2
vec_prev = pg.math.Vector2


class Agent(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.images = {'normal': pg.image.load(path.join(game.img_folder, "apple_64px.png")).convert_alpha(), \
                       'blink': pg.image.load(path.join(game.img_folder, "apple_64px_blink.png")).convert_alpha(), \
                       'wink': pg.image.load(path.join(game.img_folder, "apple_64px_wink.png")).convert_alpha()}
        self.blinks = False
        self.blink_time = .25
        self.staring_time = 3
        self.start_time = time.time()
        self.image = self.images['normal']
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.hit_rect = self.rect
        self.hit_rect.center = self.rect.center
        self.vel = vec(0, 0)
        self.position = vec(x, y)
        self.dest = vec_dest(x, y)
        self.previous_pos = vec_prev(x, y)
        self.instruction = ""
        self.orientation = "front" # left, right, front, back
        self.name = "Young Apple"
        self.silence_responses = ["can you please say that again?", "oops, I missed that. say again?", "I heard *silence*", "repeat again, please?", "could you say again?", "I didn't hear that, try again?", "I heard *silence*"]
        self.knowledge = Knowledge(self)
        self.transcript = Transcript()

        # Working memory properties
        self.recognized = []
        self.actions = [] # current, complete list of action sequences e.g. [[1],[[0],[2]]]
        self.input_to_actions = []
        self.action_queue = [] # remaining actions to be completed
        self.current_action = []
        self.key_used = ""
        #self.responses = []
        self.response = ""

    def turn(self, direction):
        """
        change the orientation of the agent to a different direction
        """
        # self.image.blit(self.img_0/90/180/270, ((x, y)))
        pass

    def give_name(self, new_name):
        self.name = new_name
        mapped_meaning = self.knowledge.lexicon()["you"]
        self.knowledge.add_to_lexicon(new_name, mapped_meaning)

    def blink(self):
        """
        Changes the apple's image to make the agent blink. 
        """
        end_time = time.time()
        elapsed = end_time - self.start_time
        if not self.blinks and elapsed > self.staring_time:
            self.image = self.images['blink']
            self.blinks = True
            self.start_time = end_time
        elif self.blinks and elapsed > self.blink_time:
            self.image = self.images['normal']
            self.blinks = False
            self.start_time = end_time

    def move_if_clear_path(self):
        """
        Checks whether the agent can continue moving to its destination 
        on a clear x and y path. If clear, moves the agent closer to its destination.
        """
        #TODO: adjust math.isclose to also check for x and y board limit value?
        clear_path = not math.isclose(self.position.x, self.dest.x, rel_tol=1e-09, abs_tol=0.5) or \
                     not math.isclose(self.position.y, self.dest.y, rel_tol=1e-09, abs_tol=0.5)
        no_walls = True

        if clear_path:
            self.knowledge.set_direction()
            #print(self.position, self.dest)
            self.position += self.vel * self.game.dt
            self.hit_rect.centerx = self.position.x
            walls_x = collide_with_walls(self, self.game.walls, 'x')
            self.hit_rect.centery = self.position.y
            walls_y = collide_with_walls(self, self.game.walls, 'y')
            self.rect.center = self.hit_rect.center
        
            if walls_x or walls_y:
                no_walls = False
                #print("walls: " + str(self.position) + ", " + str(self.dest))

        #if clear_path and no_walls:
            #printif("all clear: " + str(self.position) + ", " + str(self.dest))

        #print("checked for clear path: "+str(clear_path))
        return clear_path and no_walls
            
    def listen(self):
        '''
        Listens for a speech command, while either the 'SPACE' key or 'M' key is pressed.
        If given, command is stored in self.instruction property of the agent.
        '''
        #UNCOMMENT FOR SPEECH VERSION
        keys = pg.key.get_pressed()
        if keys[pg.K_SPACE]:
            self.key_used = "SPACE"
            self.action_queue = []
            self.response = ''
            self.vel = vec(0, 0)
            self.dest = vec_dest(self.position.x, self.position.y)
            with sr.Microphone() as source:
                try:
                    audio = r.listen(source, timeout=5)
                    self.instruction = r.recognize_google(audio).lower()
                    printif("\nYou: " + str(self.instruction))
                except:
                    self.instruction = ''
                    self.response = random.choice(self.silence_responses)
                    printif("\nYou: *silence*")
                    printif("(Hm? Can you please say that again?)")

        elif keys[pg.K_m]:
            self.key_used = "M"
            self.action_queue = []
            self.vel = vec(0, 0)
            self.dest = vec_dest(self.position.x, self.position.y)
            with sr.Microphone() as source:
            # call STT (speech to text) class to get the wav file to predict
                printif("listening...")
                try:
                    audio = r.listen(source, timeout=5)
                    self.game.morgan_speech.saveAudio(audio)
                    self.instruction = self.game.morgan_speech.getTranscription().lower()
                    printif("You: " + str(self.instruction))
                except:
                    self.response = random.choice(self.silence_responses)
                    printif("Hm? Can you please say that again?")
                    self.instruction = ''

        # ## TEXT-ONLY INPUT
        # self.instruction = input("\nType something: ").lower()
        # attempt = self.attempt()
        # printif(self.name + ": " + str(attempt))

        return self.instruction

    def interpret(self):
        """
        The Agent processes the instruction (temporarily stored in self) into
        1) words from its lexicon and learned phrases
        2) a list of actions to carry out
        """
        recognized = []
        actions = []
        unknowns = ""
        instruction = self.instruction # the input string from the user

        instruction_split = instruction.split()  # split sentence into list of words
        lexicon = self.knowledge.lexicon()
        learned = self.knowledge.learned()
        instruction_minus_phrases = instruction

        # First check for learned phrases
        for phrase in learned:
            if phrase in instruction:
                printif("found the phrase: " + str(phrase))
                recognized.append(phrase)
                actions.append(learned[phrase])

                # If found, remove phrase from instruction
                instruction_minus_phrases = instruction.replace(phrase, " ")

        instruction_split = instruction_minus_phrases.split()

        # Then check for remaining recognized words in the lexicon
        for word in instruction_split:
            if word in lexicon:
                recognized.append(word)
                actions.append(lexicon[word])

        self.recognized = recognized
        self.actions = actions
        self.input_to_actions = [(r, a) for r, a in zip(self.recognized, self.actions)]

        printif("recognized: " + str(self.recognized) + "\n action list: " + str(self.actions))

        return (recognized, actions)

    def compose_actions(self, actions):
        """
        Currently composes actions into list of single actions. Returns list.
        (Here is where semantics are helpful...:))
        Composes the actions into a meaningful sequence. 
        Returns the composed action sequence (a list of integer lists) e.g. [[1],[0],[3]]
        """
        single_actions = []
        printif("composing actions... " + str(actions))
        for action_list in actions:
            for action in action_list:            
                single_actions.append([action]) # note that action is still inside a list e.g. [1]

        # Remove move action if a destination action is given 
        # TODO: have this make use of Action object type properties 
        move = 0
        destinations = [1,2,3,4,7,9,10]

        if [move] in single_actions: # move function
            for action in single_actions:
                if action[0] in destinations: # destination function
                    while [move] in single_actions:
                        single_actions.remove([move])

        me = 8
        yes = 5
        if [me] in single_actions and [yes] in single_actions:
            while [me] in single_actions:
                single_actions.remove([me])
        
        printif("composed: " + str(single_actions))
        return single_actions

    def store_action_queue(self):
        '''
        Stores the current parsed actions into the action queue. 
        First composes actions into single list of actions e.g. from [[[0], [1]], [[1]]] to [[0],[1]]
        '''
        self.action_queue = self.compose_actions(self.actions)
        printif("stored action queue: " + str(self.action_queue))
        return self.action_queue

    def compose_feedback(self):
        """
        Temporary basic text feedback version below.
        TODO: Composes feedback into input-based response.
        """
        single_actions = self.action_queue
        input_to_actions = self.input_to_actions

        responses = ""

        for phrase, actions in input_to_actions:
            if len(actions) > 1 or len(phrase.split()) > 1:
                responses += phrase
            else:
                action = actions[0]
                action_response = self.knowledge.actions[action](response_only=True, phrase=phrase)
                responses += action_response + " "

        # #print("- single_actions: " + str(single_actions))
        # for actions in single_actions:
        #     #print("- actions: " + str(actions))
        #     for action in actions:
        #         #print("- action: " + str(action))
        #         action_response = self.knowledge.actions[action](response_only=True)
        #         responses += action_response + " "
        
        if responses:
            self.response = responses
        else:
            # Agent responds to fully unfamiliar phrases by repeating instruction
            self.response = "how do I " + self.instruction + "?" 

        printif(self.name + ": " + str(self.response))
        return self.response

    def attempt(self):
        '''
        Attempts the first action in the queue.
        '''
        action = self.action_queue[0][0]
        action_info = [action, self.transcript.entry_number()] # allows repeated actions in new queues
        #TODO: make sure this allows for the same action twice
        if action_info != self.current_action: # keeps the agent from re-calling the current action
            self.current_action = action_info
            self.knowledge.actions[action]()

    def pop_action(self):
        '''
        Pop first action from queue.
        '''
        popped = self.action_queue.pop(0)
        printif("popped: " + str(popped))

        if len(self.action_queue) == 0:
            printif("actions completed (" + str(self.action_queue) + ")")
        
        return popped


    def give_text_feedback(self):
        textRect = pg.Rect(0, 0, 0, 0)
        font = pg.font.Font(self.game.title_font, 15)
        textSurf = font.render(self.response, True, BLACK).convert_alpha()
        textSize = textSurf.get_size()
        bubbleSurf = pg.Surface((textSize[0] * 2., textSize[1] * 2))
        textRect = bubbleSurf.get_rect()
        bubbleSurf.fill(WHITE)
        bubbleSurf.blit(textSurf, textSurf.get_rect(center=textRect.center))
        textRect.center = ((WIDTH/2), (450))
        self.game.screen.blit(bubbleSurf, textRect)
        

    def update(self):
        self.listen()
        if self.instruction and not self.action_queue:
            printif("there is an instruction and no action queue yet")
            
            # Interpret instruction
            self.interpret()
            
            # Store action queue
            self.store_action_queue()
            
            # Compose feedback into response text
            self.compose_feedback()
            
            # Save to transcript
            #self.transcript.store(self.instruction, self.knowledge.readable_actions(self.action_queue.copy()), self.response)
            self.transcript.store(self.key_used, self.instruction, self.action_queue.copy(), self.response)

            # Reset instruction
            self.instruction = ""

        if self.action_queue:
            # Attempt action in queue
            self.attempt()
    
        self.blink()
        self.rect = self.image.get_rect()
        self.rect.center = self.position

        still_moving = self.move_if_clear_path()

        #printif("still moving: " + str(still_moving))
        if not still_moving and self.action_queue:
            printif("popping action now...")
            self.pop_action()
        
        # If task completed, save the task string to the transcript
        if self.game.goal_completed:
            self.transcript.store_success(self.game.goal_completed[0])

        
        self.transcript.save()
        