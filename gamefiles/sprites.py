import pygame as pg
from settings import *
from map import collide_hit_rect
vec = pg.math.Vector2
import speech_recognition as sr


def collide_with_walls(sprite, group, dir):
    if dir == 'x':
        hits = pg.sprite.spritecollide(sprite, group, False, collide_hit_rect)
        if hits:
            #print("hits x: " + str(hits))
            if hits[0].rect.centerx > sprite.hit_rect.centerx:
                sprite.position.x = hits[0].rect.left - sprite.hit_rect.width / 2
            if hits[0].rect.centerx < sprite.hit_rect.centerx:
                sprite.position.x = hits[0].rect.right + sprite.hit_rect.width / 2
            sprite.vel.x = 0
            sprite.hit_rect.centerx = sprite.position.x
    if dir == 'y':
        hits = pg.sprite.spritecollide(sprite, group, False, collide_hit_rect)
        if hits:
            #print("hits y: " + str(hits))
            if hits[0].rect.centery > sprite.hit_rect.centery:
                sprite.position.y = hits[0].rect.top - sprite.hit_rect.height / 2
            if hits[0].rect.centery < sprite.hit_rect.centery:
                sprite.position.y = hits[0].rect.bottom + sprite.hit_rect.height / 2
            sprite.vel.y = 0
            sprite.hit_rect.centery = sprite.position.y

    return hits


class Obstacle(pg.sprite.Sprite):
    def __init__(self, game, x, y, w, h):
        self.groups = game.walls
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.rect = pg.Rect(x, y, w, h)
        self.x = x
        self.y = y
        self.rect.x = x
        self.rect.y = y

class Tree(pg.sprite.Sprite):
    def __init__(self, game, x, y, w, h, top):
        pg.sprite.Sprite.__init__(self)
        self.groups = game.all_sprites
        self.game = game
        self.rect = pg.Rect(x, y, w, h)
        self.x = x
        self.y = y
        self.rect.x = x
        self.rect.y = y
        self.tree_top = top

class Tree_top(pg.sprite.Sprite):
    def __init__(self, game, x, y, w, h):
        pg.sprite.Sprite.__init__(self)
        self.groups = game.all_sprites
        self.game = game
        self.rect = pg.Rect(x, y, w, h)
        self.x = x
        self.y = y
        self.rect.x = x
        self.rect.y = y

class Bridge(pg.sprite.Sprite):
    def __init__(self, game, x, y, w, h):
        pg.sprite.Sprite.__init__(self)
        self.groups = game.all_sprites
        self.game = game
        self.rect = pg.Rect(x, y, w, h)
        self.x = x
        self.y = y
        self.rect.x = x
        self.rect.y = y

class Tasks():
    def __init__(self, task_list, goals, index):
        self.task_list = task_list
        self.goals = goals
        self.index = index
        self.completed = []

    def check_goal_state(self, cur_state):
        for i, (name, goal, index) in enumerate(zip(self.task_list, self.goals, self.index)):
            if cur_state.colliderect(goal):
                completed_task = self.task_list.pop(i)
                self.goals.pop(i)
                self.completed.append(completed_task)
        # if agent position = goal state:
        # remove current task/goal from list
        # set next task/goal
        # do smth with the transcript