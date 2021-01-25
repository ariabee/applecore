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

r = sr.Recognizer()

TASKS = "TASKS"

#background = pygame.image.load("grass.jpg")

# Create a white screen
DISPLAYSURF = pygame.display.set_mode((400, 600))
DISPLAYSURF.fill(WHITE)


class Camera:
    def __init__(self):
        self.bgimage = pygame.image.load('grass.jpg')
        self.rectBGimg = self.bgimage.get_rect()

        self.bgY1 = 0
        self.bgX1 = 0

        self.bgY2 = self.rectBGimg.height
        self.bgX2 = self.rectBGimg.width

        self.moving_speed = 5

    def update_upwards(self):
        self.bgY1 = 0
        self.bgX2 = 0
        self.bgY1 += self.moving_speed
        self.bgY2 += self.moving_speed
        if self.bgY1 >= self.rectBGimg.height:
            self.bgY1 = -self.rectBGimg.height
        if self.bgY2 >= self.rectBGimg.height:
            self.bgY2 = -self.rectBGimg.height

    def update_downwards(self):
        self.bgY1 = 0
        self.bgX2 = 0
        self.bgY1 -= self.moving_speed
        self.bgY2 -= self.moving_speed
        if self.bgY1 <= -self.rectBGimg.height:
            self.bgY1 = self.rectBGimg.height
        if self.bgY2 <= -self.rectBGimg.height:
            self.bgY2 = self.rectBGimg.height

    def update_left(self):
        self.bgX1 += self.moving_speed
        self.bgX2 += self.moving_speed
        self.bgY1 = 0
        self.bgY2 = 0
        if self.bgX1 >= self.rectBGimg.width:
            self.bgX1 = -self.rectBGimg.width
        if self.bgX2 >= self.rectBGimg.width:
            self.bgX2 = -self.rectBGimg.width

    def update_right(self):
        self.bgX1 -= self.moving_speed
        self.bgX2 -= self.moving_speed
        self.bgY1 = 0
        self.bgY2 = 0
        if self.bgX1 <= -self.rectBGimg.width:
            self.bgX1 = self.rectBGimg.width
        if self.bgX2 <= -self.rectBGimg.width:
            self.bgX2 = self.rectBGimg.width

    def render(self):
        DISPLAYSURF.blit(self.bgimage, (self.bgX1, self.bgY1))
        DISPLAYSURF.blit(self.bgimage, (self.bgX2, self.bgY2))


class Avatar(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("apple.png")
        self.surf = pygame.Surface((200, 200))
        self.rect = self.surf.get_rect(center=(340, 420))

    def move(self):
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_UP]:
            with sr.Microphone() as source:
                audio = r.listen(source)
                try:
                    text = r.recognize_google(audio)
                    print(text)
                    if text == 'up':
                        if self.rect.top > 0:
                            self.rect.move_ip(0, -100)
                        else:
                            camera.update_upwards()
                    if text == 'down':
                        if self.rect.bottom < SCREEN_HEIGHT:
                            self.rect.move_ip(0, 100)
                        else:
                            camera.update_downwards()
                    if text == 'left':
                        if self.rect.left > 0:
                            self.rect.move_ip(-100, 0)
                        else:
                            camera.update_left()
                    if text == 'right':
                        if self.rect.right < SCREEN_WIDTH:
                            self.rect.move_ip(100, 0)
                        else:
                            camera.update_right()
                except:
                    print('Did not get that try Again')
                    text = ''


# Setting up Sprites
camera = Camera()
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

    #DISPLAYSURF.blit(background, (0, 0))
    #camera.update()
    camera.render()

    tasks = font_small.render(str(TASKS), True, RED)
    DISPLAYSURF.blit(tasks, (10, 10))

    # Moves and Re-draws all Sprites
    for entity in all_sprites:
        DISPLAYSURF.blit(entity.image, entity.rect)
        entity.move()

    pygame.display.update()
    FramePerSec.tick(FPS)