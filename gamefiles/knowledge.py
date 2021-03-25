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
        self._lexicon = {'move': [0], 'run': [0], 'go': [0], 'walk': [0], 'come': [0], \
                         'left': [1], \
						 'right': [2], \
                         'up': [3], \
                         'down': [4], \
                         'yes': [5], 'no': [6], \
						 'tree': [7], \
                         'you': [8], agent.name: [8], \
                         'back': [9] }
        
        self._learned = {} # An initially empty list of learned commands mapped to actions.

        self.actions = [self.move, self.left, self.right, self.up, self.down, self.yes, self.no, self.tree, self.me, self.previous, self.climb_tree]
        self.objects = {'tree': vec(self.agent.game.tree_trunk.x, self.agent.game.tree_trunk.y), \
                        'me': agent.position, \
                        'treetop': vec(self.agent.game.tree_top.x, self.agent.game.tree_top.y)} # vector of x, y posiiton on map
        # self.confirmations = [self.yes, self.no]
        # self.categories = {"action": self.actions, "object": self.objects, "confirm": self.confirmations}

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
        #print("~~learned: " + str(self._learned))
        self.agent.response = "I learned to: " + str(words) # TODO: make this a return string instead, test it

    def link_prev_command(self):
        prior_input, prior_actions = self.agent.transcript.previous()
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
                random_coords = vec(randint(0, self.agent.game.map.width), randint(0, self.agent.game.map.height))
                self.agent.dest = random_coords
                #self.agent.response = "moving somewhere"
                return("moving somewhere")
          
    def left(self, response_only=False):
        if response_only:
            return("going left")
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            self.agent.dest.x -= 100
            #self.agent.response = "Going left..."
            return("going left")

    def right(self, response_only=False):
        if response_only:
            return("going right")
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            self.agent.dest.x += 100
            #self.agent.response = "Going right..."
            return("going right")

    def up(self, response_only=False):
        if response_only:
            return("going up")
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            self.agent.dest.y -= 100
            #self.agent.response = "Going up..."
            return("going up")

    def down(self, response_only=False):
        if response_only:
            return("going down")
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            self.agent.dest.y += 100
            #self.agent.response = "Going down..."
            return("going down")

    def yes(self, response_only=False):
        if response_only:
            response = self.link_prev_command() if not self.agent.transcript.is_empty() else ""
            return("yes! " + str(response))
        else:
            # TODO: make this increase the weight of the action for a previous command?
            response = self.link_prev_command() if not self.agent.transcript.is_empty() else ""
            #self.agent.response = "yes! " + str(response)
            return("yes! " + str(response))

    def no(self, response_only=False):
        if response_only:
            return("oops :(")
        else:
            # TODO: make this decrease the weight of the action for a previous command?
            #self.agent.response = "oops :("
            return("oops :(")

    def tree(self, response_only=False):
        if response_only:
            return "I'm going to the tree..." # Return tree vector coordinates
        else:
            self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
            tree_coords = self.objects['tree']
            self.agent.dest = tree_coords
            return self.objects['tree'] # Return tree vector coordinates

    def me(self, response_only=False):
        if response_only:
            return "me" # Return agent vector coordinates
        else:
            return self.objects['me'] # Return agent vector coordinates

    def previous(self, response_only=False):
        if response_only:
            return "I'm going back..."
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
        self.agent.previous_pos = vec(self.agent.position.x, self.agent.position.y)
        top_of_tree = self.objects['treetop']
        self.agent.dest = top_of_tree
        #return self.objects['treetop']


        #self.agent.position = vec(self.agent.game.tree_top.x, self.agent.game.tree_top.y)

        return "climbing the tree"




    
