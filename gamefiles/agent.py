import pygame as pg
from settings import *
from map import collide_hit_rect
vec = pg.math.Vector2
import speech_recognition as sr
#from knowledge import Knowledge
#from transcript import Transcript

def collide_with_walls(sprite, group, dir):
    if dir == 'x':
        hits = pg.sprite.spritecollide(sprite, group, False, collide_hit_rect)
        if hits:
            if hits[0].rect.centerx > sprite.hit_rect.centerx:
                sprite.pos.x = hits[0].rect.left - sprite.hit_rect.width / 2
            if hits[0].rect.centerx < sprite.hit_rect.centerx:
                sprite.pos.x = hits[0].rect.right + sprite.hit_rect.width / 2
            sprite.vel.x = 0
            sprite.hit_rect.centerx = sprite.pos.x
    if dir == 'y':
        hits = pg.sprite.spritecollide(sprite, group, False, collide_hit_rect)
        if hits:
            if hits[0].rect.centery > sprite.hit_rect.centery:
                sprite.pos.y = hits[0].rect.top - sprite.hit_rect.height / 2
            if hits[0].rect.centery < sprite.hit_rect.centery:
                sprite.pos.y = hits[0].rect.bottom + sprite.hit_rect.height / 2
            sprite.vel.y = 0
            sprite.hit_rect.centery = sprite.pos.y

class Agent(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.img = pg.image.load("img/avatar.png")
        self.rect = self.image.get_rect()
        self.hit_rect = self.rect
        self.hit_rect.center = self.rect.center
        #self.vx, self.vy = 0, 0
        self.vel = vec(0, 0)
        #self.x = x * TILESIZE
        #self.y = y * TILESIZE
        self.pos = vec(x, y)
        self.image.fill(RED)
        #self.image.blit(self.img, ((x, y)))
        #self.knowledge = Knowledge()
        self.instruction = ""
        self.orientation = "front" # left, right, front, back

        self.name = "Young Apple"
        #self.position = (20, 20)  # testing position
        #self.knowledge = Knowledge(self)
        #self.transcript = Transcript()
        self.current_actions = []  # working memory

    def turn(self, direction):
        """
        change the orientation of the agent to a different direction
        """
        # self.image.blit(self.img_0/90/180/270, ((x, y)))
        pass

    def get_keys(self):
        keys = pg.key.get_pressed()
        if keys[pg.K_SPACE]:
            self.listen()

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

    def climb_tree(self):
        # if standing in front of the trunk
        # just climb the tree
        pass

    def listen(self):
        # speech input
        # self.command
        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                self.instruction = r.recognize_google(audio)
                print(self.instruction)
            except:
                self.instruction = ''
                print("silence")

    def update(self):
        self.get_keys()
        self.move()
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.pos += vec(self.vx, self.vy) * self.game.dt
        self.hit_rect.centerx = self.pos.x
        collide_with_walls(self, self.game.walls, 'x')
        self.hit_rect.centery = self.pos.y
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