# ~KNOWLEDGE v.0 : every lexeme maps to a function, no separate lexeme categories/types~
# Knowledge base class for game character.
# Contains background lexicon and movement functions.
# Contains mapping and updating functions of meaning 
# to pre-existing knowledge.

from transcript import Transcript
from settings import *
from random import randint
import math
import pygame as pg

vec = pg.math.Vector2

class Knowledge:

    def __init__(self, agent):
        self.agent = agent
        # TODO: might need to make every word have a function. or a value of category, index in category
        # TODO: make action events object instances of a class
        self._lexicon = {'move': [0], 'run': [0], 'go': [0], 'walk': [0], 'come': [0], \
                         'left': [1], \
						 'right': [2], \
                         'up': [3], \
                         'down': [4], \
                         'yes': [5], 'no': [6], 'bad': [6], \
						 'tree': [7], \
                         'you': [8], agent.name: [8], \
                         'back': [9],
                         'bridge': [10],
                         'beautiful': [14], 'nice': [14], 'good': [14], 'love': [14], 'cute': [14], 'great': [14]}
        
        self._learned = {} # An initially empty list of learned commands mapped to actions.

        self.actions = [self.move, self.left, self.right, self.up, self.down, self.yes, self.no, 
                        self.tree, self.me, self.previous, self.bridge, 
                        self.climb_tree, self.cross_bridge, self.find_flowers, self.compliment]
        self.objects = {'tree': vec(self.agent.game.tree_trunk.x, self.agent.game.tree_trunk.y),
                        'me': agent.position,
                        'bridge': vec(self.agent.game.bridge.x, self.agent.game.bridge.y),
                        'treetop': vec(self.agent.game.tree_top.x, self.agent.game.tree_top.y), 
                        'bridge_crossed': vec(self.agent.game.bridge_crossed.x, self.agent.game.bridge_crossed.y), 
                        'flowers': vec(self.agent.game.red_flowers.x, self.agent.game.red_flowers.y)}
        
        # self.action_key = self.init_action_key()

    def lexicon(self):
        return self._lexicon

    def learned(self):
        return self._learned

    def add_to_lexicon(self, word, action):
        self._lexicon.update({word.lower() : action})
        # might change action to self._actions[action]
        #print(self._lexicon)
    
    def add_to_learned(self, words, action_sequence):
        """
        Add a phrase and its learned list of actions to learned.
        E.g.: "go up there", [[0],[1],[3]] stored as "go up there" : [0,1,3]
        """
        self._learned.update({words : [a[0] for a in action_sequence]}) # [0, 3, 1, 2]
        printif("~~learned: " + str(self._learned))
        return("I learned to: " + str(words))

    def link_prev_command(self):
        prior_input, prior_actions = self.agent.transcript.previous()
        printif("prior input, actions: " + str(prior_input) + ", " + str(prior_actions))
        response = self.add_to_learned(prior_input, prior_actions) 
        return response
          

    def set_direction(self):
        """
        based on current position of the agent and its destination coordinates, determine 
        in which direction (x, -x, y, -y) to move to reach the destination effectively.
        """
        difference = self.agent.dest - self.agent.position
        self.agent.vel.x, self.agent.vel.y = 0, 0
        if not math.isclose(difference.x, 0, rel_tol=1e-09, abs_tol=0.5):
            if difference.x > 0:
                self.agent.vel.x = AGENT_SPEED
            else:
                self.agent.vel.x = - AGENT_SPEED
        if not math.isclose(difference.y, 0, rel_tol=1e-09, abs_tol=0.5):
            if difference.y > 0:
                self.agent.vel.y = AGENT_SPEED
            else:
                self.agent.vel.y = - AGENT_SPEED
        # self.agent.vel.x *= 0.7071
        # self.agent.vel.y *= 0.7071
        self.agent.vel.x *= 0.5
        self.agent.vel.y *= 0.5
        
      
    def move(self, destination=None, response_only=False):
        """
        moves in random direction
        """
        if response_only:
            return("moving somewhere")
        else:
            if destination:
                self.agent.dest = destination
            elif self.agent.dest != self.agent.position:
                pass
            else:
                #TODO: update vec to be in a smaller radius/square relative to agent position 
                x, y = int(self.agent.position.x), int(self.agent.position.y)
                random_coords = vec( randint(x-40, x+40), randint(y-40, y+40) )
                #random_coords = vec(randint(0, self.agent.game.map.width), randint(0, self.agent.game.map.height))
                self.agent.dest = random_coords
                return("moving somewhere")
          
    def left(self, response_only=False):
        if response_only:
            return("going left")
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            self.agent.dest.x -= 100
            return("going left")

    def right(self, response_only=False):
        if response_only:
            return("going right")
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            self.agent.dest.x += 100
            return("going right")

    def up(self, response_only=False):
        if response_only:
            return("going up")
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            self.agent.dest.y -= 100
            return("going up")

    def down(self, response_only=False):
        if response_only:
            return("going down")
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            self.agent.dest.y += 100
            return("going down")

    def yes(self, response_only=False):
        response = "yes! "
        if response_only:
            if not self.agent.transcript.is_empty():
                prior_input, prior_actions = self.agent.transcript.previous()
                response += "I learned to: " + str(prior_input)
            return response
        else:
            # TODO: make this increase the weight of the action for a previous command?
            response = self.link_prev_command() if not self.agent.transcript.is_empty() else ""
            return("yes! " + str(response))

    def no(self, response_only=False):
        if response_only:
            return("oops :(")
        else:
            # TODO: make this decrease the weight of the action for a previous command?
            return("oops :(")

    def tree(self, response_only=False):
        if response_only:
            return "to the tree" # Return tree vector coordinates
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            tree_coords = self.objects['tree']
            self.agent.dest = tree_coords
            return self.objects['tree'] # Return tree vector coordinates
    
    def bridge(self, response_only=False):
        if response_only:
            return "I'm going to the bridge..."
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            bridge_coords = self.objects['bridge']
            self.agent.dest = bridge_coords
            return self.objects['bridge'] # Return bridge vector coordinates

    def me(self, response_only=False):
        if response_only:
            return "me" # Return agent vector coordinates
        else:
            return self.objects['me'] # Return agent vector coordinates

    def previous(self, response_only=False):
        if response_only:
            return "I'm going back"
        else:
            #print("current pos: " + str(self.agent.position) + ", dest: " +str(self.agent.previous_pos))
            previous = vec(self.agent.previous_pos.x, self.agent.previous_pos.y)    
            self.agent.dest = previous
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            return previous # Return previous agent vector coordinates
          
    # def an_object(self, object_name):
    #     coordinates = self.objects[object_name]
    #     return coordinates

    # Define complex actions / game tasks:

    def climb_tree(self, response_only=False):
    # def climb_the_tree(self, position, action_sequence):
        # (should perform actions, either predefined or passed into the function)
        # move agent to position
        # for action in action_sequence:
        #     do action

        # IDEA: could pass in action squence parameter made up of agent position + actions taken by agent from 
        # last completed action sequence (marked success)

        # if standing in front of the trunk
        # just climb the tree

        if response_only:
            return "climbing the tree"  # Return tree vector coordinates
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            treetop_coords = self.objects['treetop']
            self.agent.dest = treetop_coords
            return self.objects['treetop']  # Return treetop vector coordinates

    def cross_bridge(self, response_only=False):
        if response_only:
            return "crossing the bridge"
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            crossed_coords = self.objects['bridge_crossed']
            self.agent.dest = crossed_coords
            return self.objects['bridge_crossed']  # Return bridge crossed vector coordinates

    def find_flowers(self, response_only=False):
        if response_only:
            return "finding red flowers"
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            flower_coords = self.objects['flowers']
            self.agent.dest = flower_coords
            return self.objects['flowers']  # Return flower vector coordinates

    def compliment(self, response_only=False, phrase=""):
        if response_only:
            #return "thanks, you're " + phrase
            return "thank you"
        else:
            return "thank you"

    # #IN PROGRESS
    # def init_action_key(self):
    #     KEY = "self.move, self.left, self.right, self.up, self.down, self.yes, self.no, \
    #            self.tree, self.me, self.previous, self.bridge, self.climb_tree, self.cross_bridge,self.find_flowers"
    #     #KEY = KEY.replace("self.", "").replace("[", "").replace("]", "")
    #     KEY = KEY.replace("self.", "")
    #     KEY = KEY.split(", ")

    #     printif("KEY initialized: " + str(KEY))
    #     return KEY 

    # def readable_actions(self, list_of_action_ints):
    #     KEY = self.action_key
    #     action_names = []

    #     for action_int in list_of_action_ints:
    #         action_names.append(KEY[action_int[0]])

    #     printif("readable actions: " + str(action_names))
    #     return action_names





    
