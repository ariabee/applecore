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

vec = pg.math.Vector2
vec_dest = pg.math.Vector2
vec_prev = pg.math.Vector2


class Agent(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.images = [pg.image.load(path.join(game.img_folder, "apple_64px.png")), \
                       pg.image.load(path.join(game.img_folder, "apple_64px_blink.png"))]
        self.blinks = False
        self.blink_time = .25
        self.staring_time = 3
        self.start_time = time.time()
        self.image = self.images[0]
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
        self.knowledge = Knowledge(self)
        self.transcript = Transcript()

        # Working memory properties
        self.recognized_words = []
        self.actions = [] # current, complete list of action sequences e.g. [[1],[[0],[2]]]
        self.action_queue = [] # remaining actions to be completed
        self.current_action = []
        #self.responses = []
        self.response = ""
        self.tasks = ["Go to the tree!"]

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
            self.image = self.images[1]
            self.blinks = True
            self.start_time = end_time
        elif self.blinks and elapsed > self.blink_time:
            self.image = self.images[0]
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
            self.action_queue = []
            self.vel = vec(0, 0)
            self.dest = vec_dest(self.position.x, self.position.y)
            with sr.Microphone() as source:
                try:
                    audio = r.listen(source, timeout=5)
                    self.instruction = r.recognize_google(audio)
                    printif("\nYou: " + str(self.instruction))
                except:
                    self.instruction = ''
                    printif("\nYou: *silence*")
                    printif("(Hm? Can you please say that again?)")
            # attempt = self.attempt()
            # printif(self.name + ": " + str(attempt))

        elif keys[pg.K_m]:
            self.action_queue = []
            self.vel = vec(0, 0)
            self.dest = vec_dest(self.position.x, self.position.y)
            with sr.Microphone() as source:
            # call STT (speech to text) class to get the wav file to predict
                printif("listening...")
                try:
                    audio = r.listen(source, timeout=5)
                    self.game.morgan_speech.saveAudio(audio)
                    self.instruction = self.game.morgan_speech.getTranscription()
                    printif("You: " + str(self.instruction))
                except:
                    printif("Hm? Can you please say that again?")
                    self.instruction = ''
            # attempt = self.attempt()
            # printif(self.name + ": " + str(attempt))

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
        return: composition, the recognized string of words
        return: actions, the list of corresponding actions
        """
        recognized_words = ""
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
                recognized_words += (phrase + " ")
                actions.append(learned[phrase])

                # If found, remove phrase from instruction
                instruction_minus_phrases = instruction.replace(phrase, " ")

        instruction_split = instruction_minus_phrases.split()

        # Then check for remaining recognized words in the lexicon
        for word in instruction_split:
            if word in lexicon:
                recognized_words += (word + " ")
                actions.append(lexicon[word])
                # print(actions)

        self.recognized_words = recognized_words
        self.actions = actions

        return (recognized_words, actions)

    def store_action_queue(self):
        '''
        Stores the current parsed actions into the action queue. 
        First composes actions into single list of actions e.g. from [[[0], [1]], [[1]]] to [[0],[1]]
        '''
        self.action_queue = self.compose_actions(self.actions)
        printif("stored action queue: " + str(self.action_queue))
        return self.action_queue

    def compose_actions(self, actions):
        """
        Currently composes actions into list of single actions. Returns list.
        (Here is where semantics are helpful...:))
        Composes the actions into a meaningful sequence. 
        Returns the composed action sequence (a list of integer lists) e.g. [[1],[0,3,1],[2]]
        """
        single_actions = []
        for action_list in actions:
            for action in action_list:
                single_actions.append([action]) # note that action is still inside a list e.g. [1]
        return single_actions

    def compose_feedback(self):
        """
        Temporary basic text feedback version below.
        TODO: Composes feedback into input-based response.
        """
        single_actions = self.action_queue

        responses = ""

        for actions in single_actions:
            for action in actions:
                action_response = self.knowledge.actions[action](response_only=True)
                responses = action_response + ". "
        
        self.response = responses

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

    def try_actions(self, action_queue):
        """
        Execute the retrieved action functions.
        """
        responses = []

        for actions in action_queue:
            for action in actions:
                action_response = self.knowledge.actions[action]()  # do the action, get the response

                responses.append(action_response)  # todo: update responses to be specific to user's words
                                                   # (through a function parameter or through function itself)

        return (responses)

    def attempt_old(self):
        """
        Make an attempt to interpret and parse actions from the given input.
        return: compositon, the composed and recognized string of words
        return: responses, the responses generated by doing the actions
        """
        # 1) Interpret the instruction
        composition, parsed_actions = self.interpret()

        # 2) Compose the actions into a meaningful sequence and save to working memory
        self.action_queue = self.compose_actions(parsed_actions)
        
        # 3) Save the instruction and current actions to transcript
        if self.instruction:
            self.transcript.store(self.instruction, self.action_queue)

        # 4) Try the actions and collect the responses
        responses = self.try_actions(self.action_queue)
        #responses = ""

        return (composition, responses)

    # def try_action(self):
    #     """
    #     Execute the next action in the queue and return the response.
    #     """
    #     response = ""
    #     if self.action_queue:
    #         action = self.action_queue[0][0]
    #         response = self.knowledge.actions[action]()
    #         print("next action: " + str(action) + ", response: " + str(response))
        
    #     return response

    # def try_action_in_queue(self):
    #     """
    #     """
    #     responses = []
    #     if self.action_queue: # if there are still actions to complete
    #         print("action_queue: " + str(self.action_queue))
    #         action = self.action_queue[0][0]
    #         response = self.knowledge.actions[action]() # do action, store the response
    #         print("action: " + str(action) + ", response: " + str(response))

    #         still_moving = self.move_if_clear_path() # continue completing the action
            
    #         if not still_moving: # if the action is complete, add the response and pop the action off the queue
    #             responses.append(response)
    #             print("responses: " + str(responses))
    #             popped = self.action_queue.pop(0)
    #             print("popped: " + str(popped))

    def display_tasks(self):
        textRect = pg.Rect(0, 0, 0, 0)
        font = pg.font.Font(self.game.title_font, 15)
        height = 0
        for task in self.tasks:
            textSurf = font.render(task, True, BLACK).convert_alpha()
            textSize = textSurf.get_size()
            height += textSize[0]
            bubbleSurf = pg.Surface((textSize[0] * 2., textSize[1] * 2))
            textRect = bubbleSurf.get_rect()
            bubbleSurf.fill(LIGHTGREY)
            bubbleSurf.blit(textSurf, textSurf.get_rect(center=textRect.center))
            textRect.center = ((700), (height))
            self.game.screen.blit(bubbleSurf, textRect)


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
            printif("there is an instruction and no action queue")
            
            # Interpret instruction
            self.interpret()
            
            # Store action queue
            self.store_action_queue()
            
            # Compose feedback into response text
            self.compose_feedback()
            
            # Save to transcript
            self.transcript.store(self.instruction, self.action_queue)
            
            # Reset instruction
            self.instruction = ""
            # TODO: update transcript.py so that: self.transcript.store(self.instruction, self.action_queue, self.response)

        # Attempt action in queue
        if self.action_queue:
            self.attempt()
    
        self.blink()
        self.rect = self.image.get_rect()
        self.rect.center = self.position

        # response = self.try_action()
        # still_moving = self.move_if_clear_path()
        # if not still_moving and self.action_queue:
        #     self.responses.append(response)
        #     self.action_queue.pop(0)

        still_moving = self.move_if_clear_path()
        #printif("still moving: " + str(still_moving))
        if not still_moving and self.action_queue:
            printif("pop action now")
            self.pop_action()

        self.transcript.save()
        