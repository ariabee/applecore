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

vec = pg.math.Vector2
vec_dest = pg.math.Vector2


class Agent(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.image.load(path.join(game.img_folder, "apple_64px.png"))
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.hit_rect = self.rect
        self.hit_rect.center = self.rect.center
        self.vel = vec(0, 0)
        self.position = vec(x, y)
        self.dest = vec_dest(x, y)
        self.instruction = ""
        self.orientation = "front" # left, right, front, back
        self.name = "Young Apple"
        self.knowledge = Knowledge(self)
        self.transcript = Transcript()
        self.current_actions = []  # working memory
        self.response = ""
        self.tasks = ["Climb the tree!", "TEST", "TEST"]
        self.previous_pos = self.position


    def turn(self, direction):
        """
        change the orientation of the agent to a different direction
        """
        # self.image.blit(self.img_0/90/180/270, ((x, y)))
        pass

    def listen_attempt(self):
        #UNCOMMENT FOR SPEECH VERSION
        keys = pg.key.get_pressed()
        if keys[pg.K_SPACE]:
            self.vel = vec(0, 0)
            self.dest = vec_dest(self.position.x, self.position.y)
            with sr.Microphone() as source:
                audio = r.listen(source)
                try:
                    self.instruction = r.recognize_google(audio)
                    print("\nYou: " + str(self.instruction))
                except:
                    self.instruction = ''
                    print("\nYou: *silence*")
                    print("(Hm? Can you please say that again?)")
            attempt = self.attempt()
            print(self.name + ": " + str(attempt))

        # ## TEXT-ONLY INPUT
        # self.instruction = input("\nType something: ").lower()
        # attempt = self.attempt()
        # print(attempt)

        elif keys[pg.K_m]:
            self.vel = vec(0, 0)
            self.dest = vec_dest(self.position.x, self.position.y)
            with sr.Microphone() as source:
                audio = r.listen(source)
            # call STT (speech to text) class to get the wav file to predict
                try:
                    input = self.game.morgan_speech.inputLoad(audio)
                    self.instruction = self.game.morgan_speech.get_prediction(input)
                    print("You: " + str(self.instruction))
                except:
                    print("Hm? Can you please say that again?")
                    self.instruction = ''
            attempt = self.attempt()
            print(self.name + ": " + str(attempt))

        return self.instruction


    def give_name(self, new_name):
        self.name = new_name
        mapped_meaning = self.knowledge.lexicon()["you"]
        self.knowledge.add_to_lexicon(new_name, mapped_meaning)

    def store_parsed_actions(self, parsed_actions):
        self.current_actions = parsed_actions

    def interpret(self):
        """
        The Agent processes the instruction (temporarily stored in self) into
        1) words from its lexicon and learned phrases
        2) a list of actions to carry out
        return: composition, the recognized string of words
        return: actions, the list of corresponding actions
        """

        composition = ""
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
                print("found the phrase: " + str(phrase))
                composition += (phrase + " ")
                actions.append(learned[phrase])

                # If found, remove phrase from instruction
                instruction_minus_phrases = instruction.replace(phrase, " ")

        instruction_split = instruction_minus_phrases.split()

        # Then check for remaining recognized words in the lexicon
        for word in instruction_split:
            if word in lexicon:
                composition += (word + " ")
                actions.append(lexicon[word])
                # print(actions)

        # self.store_parsed_actions(actions)

        return (composition, actions)

    def compose_actions(self, actions):
        """
        (Here is where semantics are helpful...:))
        Composes the actions into a meaningful sequence. 
        Returns the composed action sequence (a list of integer lists) e.g. [[1],[0,3,1],[2]]
        """
        # single_actions = []

        # for action_list in actions:
        #     if len(action_list)==1:
        #         single_actions.append(action_list[0])
        
        # if 0 in single_actions:
        #     for action in single_actions:
        #         print(str(type(action)))
               

        # print("actions: " + str(actions))
        return actions

    def try_actions(self, parsed_actions):
        """
        Execute the retrieved action functions.
        """
        responses = []
        
        for actions in parsed_actions:
            for action in actions:
                action_response = self.knowledge.actions[action]()  # do the action, get the response
                responses.append(action_response)  # todo: update responses to be specific to user's words
            # (through a function parameter or through function itself)

        return (responses)

    def attempt(self):
        """
        Make an attempt to interpret and parse actions from the given input.
        return: compositon, the composed and recognized string of words
        return: responses, the responses generated by doing the actions
        """
        instruction = self.instruction

        # 1) Interpret the instruction
        composition, parsed_actions = self.interpret()

        # 2) Compose the actions into a meaningful sequence and save to working memory
        self.current_actions = self.compose_actions(parsed_actions)

        # # Save the parsed actions to working memory
        # self.current_actions = parsed_actions
        
        # 3) Save the instruction and current actions to transcript
        if self.instruction:
            self.transcript.store(self.instruction, self.current_actions)

        # 4) Try the actions and collect the responses
        responses = self.try_actions(parsed_actions)

        return (composition, responses)

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
        self.listen_attempt()
        self.rect = self.image.get_rect()
        self.rect.center = self.position
       
        #TODO: put this in the right spot so it stores the previous position before moving!
        self.previous_pos = self.position
        #print("previous_pos: " + str(self.previous_pos))

        #TODO: put the below code into a method, call the method inside update
        if not math.isclose(self.position.x, self.dest.x, rel_tol=1e-09, abs_tol=0.5) or \
                not math.isclose(self.position.y, self.dest.y, rel_tol=1e-09, abs_tol=0.5):
            self.knowledge.set_direction()
            #print(self.position, self.dest)
            self.position += self.vel * self.game.dt
            self.hit_rect.centerx = self.position.x
            collide_with_walls(self, self.game.walls, 'x')
            self.hit_rect.centery = self.position.y
            collide_with_walls(self, self.game.walls, 'y')
            self.rect.center = self.hit_rect.center

        self.transcript.save()

