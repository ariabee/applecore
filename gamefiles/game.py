# Apple Core-dination game.
# KidsCanCode - Game Development with Pygame video series
# Tile-based game - Part 4
# Scrolling Map/Camera
# Video link: https://youtu.be/3zV2ewk-IGU
import pygame as pg
import sys
from os import path
from settings import *
from sprites import *
from map import *
from agent import *

import random, time
import torch
import torchaudio
from asr.m5 import M5
from asr.speech_to_text import SpeechToText
import speech_recognition as sr


class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.load_data()

    def load_data(self):
        game_folder = path.dirname(__file__)
        self.img_folder = path.join(game_folder, "img")
        self.map_folder = path.join(game_folder, "maps")
        self.map = TiledMap(path.join(self.map_folder, "tiled_map.tmx"))
        self.map_img = self.map.make_map()
        self.map_rect = self.map_img.get_rect()
        self.title_font = path.join(self.img_folder, 'arial.ttf')

        self.load_asr()

    def load_asr(self):
        # initialize path to the wav file to be predicted
        path_to_wav = "user_input.wav"

        # initialize device for cpu or gpu
        use_cuda = torch.cuda.is_available()
        torch.manual_seed(7)
        device = torch.device("cuda" if use_cuda else "cpu")

        # resample the wav files from 1600 to 8000
        sample_rate = 1600
        new_sample_rate = 8000
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

        # initialize model, M5, with proper parameters
        model = M5(n_input=1, n_output=35)

        # initialize path to local (local machine) model
        path_to_local_model = "speech_commands_model/speech_commands_model.pt"

        # load trained model
        model.load_state_dict(torch.load(path_to_local_model))


    def new(self):
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        for tile_object in self.map.map_data.objects:
            if tile_object.name == "game_border":
                self.game_border = Obstacle(self, tile_object.x, tile_object.y, tile_object.width, tile_object.height)
            if tile_object.name == "water":
                self.water = Obstacle(self, tile_object.x, tile_object.y, tile_object.width, tile_object.height)
            if tile_object.name == "tree_top":
                self.tree_top = Tree_top(self, tile_object.x, tile_object.y, tile_object.width, tile_object.height)
            if tile_object.name == "tree_trunk":
                self.tree_trunk = Tree(self, tile_object.x, tile_object.y, tile_object.width, tile_object.height,
                                           self.tree_top)
        for tile_object in self.map.map_data.objects:
            if tile_object.name == "agent":
                self.agent = Agent(self, tile_object.x, tile_object.y)
        self.camera = Camera(self.map.width, self.map.height)

    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def quit(self):
        pg.quit()
        sys.exit()

    def update(self):
        # update portion of the game loop
        self.all_sprites.update()
        self.camera.update(self.agent)

    def draw_grid(self):
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw_text(self, text, font_name, size, color, x, y, align="topleft"):
        font = pg.font.Font(font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(**{align: (x, y)})
        self.screen.blit(text_surface, text_rect)

    def draw(self):
        self.screen.blit(self.map_img, self.camera.apply_rect(self.map_rect))
        for sprite in self.all_sprites:
            self.screen.blit(sprite.image, self.camera.apply(sprite))
        pg.display.flip()

    def wait_for_key(self):
        pg.event.wait()
        waiting = True
        while waiting:
            self.clock.tick(FPS)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    waiting = False
                    self.quit()
                if event.type == pg.KEYUP:
                    waiting = False

    def events(self):
        # catch all events here
        # speech input
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()

    def show_start_screen(self):
        #self.intro()
        self.screen.fill(DARKGREEN)
        self.draw_text("Hello, and welcome to the world of me, Young Apple.", self.title_font, 35, WHITE, WIDTH / 2,
                       HEIGHT / 2, align="center")
        self.draw_text("I'm ready to move around and learn new tricks.", self.title_font, 35, WHITE, WIDTH / 2,
                       HEIGHT * 2 / 3, align="center")
        self.draw_text("Press a key to start", self.title_font, 20, WHITE,
                       WIDTH / 2, HEIGHT * 3 / 4, align="center")
        self.screen.blit(pg.image.load(path.join(self.img_folder, "apple_64px.png")), (WIDTH / 2, 100))
        pg.display.flip()
        self.wait_for_key()

    def show_go_screen(self):
        pass

    def name_agent(self):
        '''
        User names the game agent. Returns string name of agent. 
        '''
        name = input("What would you like to call me when teaching me tricks? ")
        confirm = input("Call me, a young apple, '" + name + "'? (y/n) ")
        while confirm.lower()=="n":
            name = input("Okay, what would you like to call me? ")
            confirm = input("Call me, '" + name + "'? (y/n) ")
        print("Terrific. '" + name +"' is my name!" )

        return(name)
    
    def intro(self):
        print("\n            *******************************************************\n\
            * Hello, and welcome to the world of me, Young Apple. *\n\
            * I'm ready to move around and learn new tricks.      *\n\
            *******************************************************\n")

# create the game object
g = Game()
g.show_start_screen()

while True:
    g.new()

    # Give Young Apple a name
    name = g.name_agent()
    g.agent.give_name(name.lower())
    print("Teach me, " + name + ", to: climb the tree.")

    g.run()
    g.show_go_screen()