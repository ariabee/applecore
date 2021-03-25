import speech_recognition as sr
import pygame as pg
import sys

# Define some colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTLIGHTGREY = (239, 247, 244)
LIGHTGREY = (100, 100, 100)
GREEN = (0, 255, 0)
DARKGREEN = (0, 100, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BROWN = (106, 55, 5)
BLUE = (0, 85, 255)
CYAN = (0, 255, 255)

# Game settings
WIDTH = 800
HEIGHT = 480
FPS = 60
TITLE = "APPLE CORE-DINATION"

TILESIZE = 32
GRIDWIDTH = WIDTH / TILESIZE
GRIDHEIGHT = HEIGHT / TILESIZE

# Player settings
AGENT_SPEED = 30

r = sr.Recognizer()

# Hide torchaudio future package warning
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Debug settings
DEBUG = True

def printif(text):
    if DEBUG:
        print(str(text))