import pygame, sys
from pygame.locals import *
import random, time

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

background = pygame.image.load("grass.jpg")

# Create a white screen
DISPLAYSURF = pygame.display.set_mode((400, 600))
DISPLAYSURF.fill(WHITE)

"""
class Camera:
    def __init__(self, position):

        self.offset = Vector(windowWidth / 2, windowHeight / 2)
        self.scale = Vector(windowWidth, windowHeight)
        self.position = position - self.offset + self.level.player.scale / 2
        self.rect = pygame.Rect(self.position, self.scale)
        self.velocity = Vector(0, 0)

        self.image = pygame.Surface((10, 10)).convert()
        self.image.fill((0, 0, 255))
        camera_position = self.position + self.offset
        player_position = self.level.player.position + self.level.player.scale / 2
        distance = player_position - camera_position
        cam_x = 0  # 150 # allowed_distance_x_from_camera

    def update(self, milliseconds):
        camera_position.y = player_position.y
        if distance.x > cam_x:
            camera_position.x = player_position.x - cam_x
        if distance.x < -cam_x:
            camera_position.x = player_position.x + cam_x

        self.position = camera_position - self.offset

        self.rect.topleft = self.position  # - self.offset
"""

class Avatar(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("apple.png")
        self.surf = pygame.Surface((90, 90))
        self.rect = self.surf.get_rect(center=(340, 420))

    def move(self):
        pressed_keys = pygame.key.get_pressed()

        if self.rect.top > 0:
            if pressed_keys[K_UP]:
                self.rect.move_ip(0, -5)
        if self.rect.bottom < SCREEN_HEIGHT:
            if pressed_keys[K_DOWN]:
                self.rect.move_ip(0, 5)

        if self.rect.left > 0:
            if pressed_keys[K_LEFT]:
                self.rect.move_ip(-5, 0)
        if self.rect.right < SCREEN_WIDTH:
            if pressed_keys[K_RIGHT]:
                self.rect.move_ip(5, 0)

# Setting up Sprites
P1 = Avatar()

# Creating Sprites Groups
all_sprites = pygame.sprite.Group()
all_sprites.add(P1)

# Game Loop
while True:

    # Cycles through all events occuring
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    DISPLAYSURF.blit(background, (0, 0))
    tasks = font_small.render(str(TASKS), True, RED)
    DISPLAYSURF.blit(tasks, (10, 10))

    # Moves and Re-draws all Sprites
    for entity in all_sprites:
        DISPLAYSURF.blit(entity.image, entity.rect)
        entity.move()

    pygame.display.update()
    FramePerSec.tick(FPS)