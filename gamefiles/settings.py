import pygame as pg

vec = pg.math.Vector2

WIDTH = 1024
HEIGHT = 512
FPS = 60
TITLE = "TEST"

# Creating colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTGREY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BROWN = (106, 55, 5)

TILESIZE = 64
GRIDWITH = WIDTH / TILESIZE
GRIDHEIGHT = HEIGHT / TILESIZE
BGCOLOR = BROWN
avatar_img = 'avatar.png'
PLAYER_HIT_RECT = pg.Rect(0, 0, 35, 35)

TASKS = "TASKS"

# Layers
WALL_LAYER = 1
AVATAR_LAYER = 2