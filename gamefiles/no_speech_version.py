import pygame, sys
from pygame.locals import *
import random, time
import speech_recognition as sr

# Initialzing
pygame.init()

# Setting up FPS
FPS = 60
FramePerSec = pygame.time.Clock()

# Creating colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Setting up Fonts
font = pygame.font.SysFont("Verdana", 60)
font_small = pygame.font.SysFont("Verdana", 20)

# Other Variables for use in the program
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

TASKS = "TASKS"

# Create a white screen
DISPLAYSURF = pygame.display.set_mode((400, 600))
DISPLAYSURF.fill(WHITE)
image = pygame.image.load('grass.jpg')
DISPLAYSURF.blit(image, (0, 0))

tasks = font_small.render(str(TASKS), True, RED)

class Background():

    def __init__(self):
        self.image = pygame.image.load('grass.jpg')
        self.bgX = 0
        self.bgX2 = self.image.get_width()
        self.bgY = 0
        self.bgY2 = self.image.get_height()

    def updateX(self, steps):
        self.bgX -= steps
        self.bgX2 -= steps

        if self.bgX < self.image.get_width() * -1:
            self.bgX = self.image.get_width()
        if self.bgX > self.image.get_width():
            self.bgX = -self.image.get_width()

        if self.bgX2 < self.image.get_width() * -1:
            self.bgX2 = self.image.get_width()
        if self.bgX2 > self.image.get_width():
            self.bgX2 = -self.image.get_width()

    def updateY(self, steps):
        self.bgY -= steps
        self.bgY2 -= steps

        if self.bgY < self.image.get_height() * -1:
            self.bgY = self.image.get_height()
        if self.bgY > self.image.get_height():
            self.bgY = -self.image.get_height()

        if self.bgY2 < self.image.get_height() * -1:
            self.bgY2 = self.image.get_height()
        if self.bgY2 > self.image.get_height():
            self.bgY2 = -self.image.get_height()

    def redrawBackground(self):
        DISPLAYSURF.blit(self.image, (self.bgX, 0))
        DISPLAYSURF.blit(self.image, (self.bgX2, 0))
        DISPLAYSURF.blit(self.image, (0, self.bgY))
        DISPLAYSURF.blit(self.image, (0, self.bgY2))
        DISPLAYSURF.blit(tasks, (10, 10))

class Avatar(pygame.sprite.Sprite):

    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("apple.png")
        self.surf = pygame.Surface((200, 200))
        self.rect = self.surf.get_rect(center=(340, 420))

    def move(self):
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_UP]:
            if self.rect.top > 0:
                self.rect.move_ip(0, -5)
            else:
                background.updateY(-5)
        if pressed_keys[K_DOWN]:
            if self.rect.bottom < SCREEN_HEIGHT:
                self.rect.move_ip(0, 5)
            else:
                background.updateY(5)
        if pressed_keys[K_LEFT]:
            if self.rect.left > 0:
                self.rect.move_ip(-5, 0)
            else:
                background.updateX(-5)
        if pressed_keys[K_RIGHT]:
            if self.rect.right < SCREEN_WIDTH:
                self.rect.move_ip(5, 0)
            else:
                background.updateX(5)

# Setting up Sprites
P1 = Avatar()

# Creating Sprites Groups
all_sprites = pygame.sprite.Group()
all_sprites.add(P1)

background = Background()

# Game Loop
while True:

    #DISPLAYSURF.blit(background, (0, 0))
    background.redrawBackground()

    # Cycles through all events occurring
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    # Moves and Re-draws all Sprites
    for entity in all_sprites:
        DISPLAYSURF.blit(entity.image, entity.rect)
        entity.move()

    pygame.display.update()
    FramePerSec.tick(FPS)