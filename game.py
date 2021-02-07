import pygame, sys
from pygame.locals import *
import random, time
import torch
import torchaudio
from asr import m5
from asr.m5 import M5
from asr.speech_to_text import SpeechToText
import speech_recognition as sr

#initialize path to local (local machine) model
path_to_local_model = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/speech_commands_model/speech_commands_model.pt"

#initialize path to the wav file to be predicted
path_to_wav = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/bed.wav"

# Initialzing
pygame.init()

# Setting up FPS
FPS = 60
FramePerSec = pygame.time.Clock()

# Creating colors
WHITE = (255, 255, 255)

# Setting up Fonts
font = pygame.font.SysFont("Verdana", 60)
font_small = pygame.font.SysFont("Verdana", 20)

# Other Variables for use in the program
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

background = pygame.image.load("grass.jpg")

# Create a white screen
DISPLAYSURF = pygame.display.set_mode((400, 600))
DISPLAYSURF.fill(WHITE)

#initialize device for cpu or gpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

#resample the wav files from 1600 to 8000
sample_rate = 1600
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

#initialize model, M5, with proper parameters
model = M5(n_input=1, n_output=35)

#load trained model
model.load_state_dict(torch.load(path_to_local_model))

#call STT (speech to text) class to get the wav file to predict
user_input = SpeechToText.userInput(path_to_wav)

#call STT class to get the waveform from the user_input
waveform = SpeechToText.inputLoad(path_to_wav)

#call STT class to get a prediction on the wav file
prediction = SpeechToText.get_prediction(waveform, device, transform, model)
print(prediction)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("apple.png")
        self.surf = pygame.Surface((40, 75))
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
P1 = Player()

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

    # Moves and Re-draws all Sprites
    for entity in all_sprites:
        DISPLAYSURF.blit(entity.image, entity.rect)
        entity.move()

    pygame.display.update()
    FramePerSec.tick(FPS)