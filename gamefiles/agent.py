import pygame as pg
import os
from settings import *
from map import collide_hit_rect
vec = pg.math.Vector2
from sprites import *
import speech_recognition as sr
from knowledge import Knowledge
from transcript import Transcript


class Agent(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.Surface((TILESIZE, TILESIZE))
        # self.img = pg.image.load(os.path.join("img", "agent_32px.png"))
        # self.img = pg.image.load('img/avatar.png')
        self.rect = self.image.get_rect()
        self.hit_rect = self.rect
        self.hit_rect.center = self.rect.center
        self.vel = vec(0, 0)
        #self.x = x * TILESIZE
        #self.y = y * TILESIZE
        self.position = vec(x, y)
        self.image.fill(RED)
        #self.image.blit(self.img, ((x, y)))
        self.instruction = ""
        self.orientation = "front" # left, right, front, back

        self.name = "Young Apple"
        #self.position = (20, 20)  # testing position
        self.knowledge = Knowledge(self)
        self.transcript = Transcript()
        self.current_actions = []  # working memory

    def turn(self, direction):
        """
        change the orientation of the agent to a different direction
        """
        # self.image.blit(self.img_0/90/180/270, ((x, y)))
        pass

    def listen_attempt(self):
        # speech input
        # self.command

        ##UNCOMMENT FOR SPEECH VERSION
        # keys = pg.key.get_pressed()
        # if keys[pg.K_SPACE]:
        #     self.vel = vec(0, 0)
        #     with sr.Microphone() as source:
        #         audio = r.listen(source)
        #         try:
        #             self.instruction = r.recognize_google(audio)
        #             print("You: " + str(self.instruction))
        #         except:
        #             self.instruction = ''
        #             print("silence")
        #     attempt = self.attempt()
        #     print(self.name + ": " + str(attempt))

        ## TEXT-ONLY INPUT
        self.instruction = input("\nType something: ").lower()
        attempt = self.attempt()
        print(attempt)

        return self.instruction


        """
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            self.vx = -PLAYER_SPEED
        if keys[pg.K_RIGHT] or keys[pg.K_d]:
            self.vx = PLAYER_SPEED
        if keys[pg.K_UP] or keys[pg.K_w]:
            self.vy = -PLAYER_SPEED
        if keys[pg.K_DOWN] or keys[pg.K_s]:
            self.vy = PLAYER_SPEED
        if self.vx != 0 and self.vy != 0:
            self.vx *= 0.7071
            self.vy *= 0.7071
        """

    def give_name(self, new_name):
        self.name = new_name
        mapped_meaning = self.knowledge.lexicon()["you"]
        self.knowledge.add_to_lexicon(new_name, mapped_meaning)

    def store_parsed_actions(self, parsed_actions):
        self.current_actions = parsed_actions

    def interpret(self):
        """
        The Agent processes the instruction into
        1) words "it understands" / that are retrievable in the knowledge base
        2) a list of actions to carry out
        param: instruction, the input string from the user
        return: composition, the recognized string of words
        return: actions, the list of corresponding actions
        """

        composition = ""
        actions = []
        unknowns = ""
        instruction = self.instruction

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

        # Check for movement words in the instruction that the agent also recognizes
        for word in instruction_split:
            if word in lexicon:
                composition += (word + " ")
                actions.append(lexicon[word])
                # print(actions)

        self.store_parsed_actions(actions)

        return (composition, actions)
        # return(composition, self.try_actions(actions))

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
        param: instruction, the input string from the user
        return: compositon, the composed and recognized string of words
        return: responses, the responses generated by doing the actions
        """
        instruction = self.instruction
        # Interpret the instructions
        composition, parsed_actions = self.interpret()

        # Save the parsed actions to working memory
        self.current_actions = parsed_actions
        
        # Save the instruction and current actions to transcript
        if self.instruction:
            self.transcript.store(self.instruction, self.current_actions)

        # Try the actions and collect the responses
        responses = self.try_actions(parsed_actions)

        return (composition, responses)

    """
    def move(self):
        self.vx, self.vy = 0, 0
        if self.instruction == "left":
            self.vx = -AGENT_SPEED
        if self.instruction== "right":
            self.vx = AGENT_SPEED
        if self.instruction == "up":
            self.vy = -AGENT_SPEED
        if self.instruction == "down":
            self.vy = AGENT_SPEED
        if self.vx != 0 and self.vy != 0:
            self.vx *= 0.7071
            self.vy *= 0.7071
            """

    """
    def collide_with_walls(self, dir):
        if dir == 'x':
            hits = pg.sprite.spritecollide(self, self.game.walls, False)
            if hits:
                if self.vx > 0:
                    self.x = hits[0].rect.left - self.rect.width
                if self.vx < 0:
                    self.x = hits[0].rect.right
                self.vx = 0
                self.rect.x = self.x
        if dir == 'y':
            hits = pg.sprite.spritecollide(self, self.game.walls, False)
            if hits:
                if self.vy > 0:
                    self.y = hits[0].rect.top - self.rect.height
                if self.vy < 0:
                    self.y = hits[0].rect.bottom
                self.vy = 0
                self.rect.y = self.y
    """

    # def climb_tree(self):
    #     # if standing in front of the trunk
    #     # just climb the tree
    #     pass


    def update(self):
        self.listen_attempt()
        self.rect = self.image.get_rect()
        self.rect.center = self.position
        self.position += self.vel * self.game.dt
        self.hit_rect.centerx = self.position.x
        collide_with_walls(self, self.game.walls, 'x')
        self.hit_rect.centery = self.position.y
        collide_with_walls(self, self.game.walls, 'y')
        self.rect.center = self.hit_rect.center

    """
    old update function for old map
    def update(self):
        # put command into Knowledge
        # self.command = ""
        self.get_keys()
        # call turn function
        self.x += self.vx * self.game.dt
        self.y += self.vy * self.game.dt
        self.rect.x = self.x
        self.collide_with_walls('x')
        self.rect.y = self.y
        self.collide_with_walls('y')
        """