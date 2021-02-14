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

#import random, time
#import torch
#import torchaudio
#from asr import m5
#from asr.m5 import M5
#from asr.speech_to_text import SpeechToText
#import speech_recognition as sr


class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.load_data()
        # initialize model, M5, with proper parameters
        #self.model = M5(n_input=1, n_output=35)
        # load trained model
        #self.model.load_state_dict(torch.load(path_to_local_model))

    def load_data(self):
        game_folder = path.dirname(__file__)
        self.img_folder = path.join(game_folder, "img")
        self.map_folder = path.join(game_folder, "maps")
        #self.map = Map(path.join(map_folder, "map.txt"))
        self.map = TiledMap(path.join(self.map_folder, "tiled_map.tmx"))
        self.map_img = self.map.make_map()
        self.map_rect = self.map_img.get_rect()

        # initialize path to local (local machine) model
        path_to_local_model = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/speech_commands_model/speech_commands_model.pt"

        # initialize path to the wav file to be predicted
        path_to_wav = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/bed.wav"

        # initialize device for cpu or gpu
        #use_cuda = torch.cuda.is_available()
        #torch.manual_seed(7)
        #device = torch.device("cuda" if use_cuda else "cpu")

        # resample the wav files from 1600 to 8000
        sample_rate = 1600
        new_sample_rate = 8000
        #transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

    """
    old new function which loads an old map
    def new(self):
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        for row, tiles in enumerate(self.map.data):
            for col, tile in enumerate(tiles):
                if tile == 'A':
                    Water(self, col, row)
                if tile == 'W':
                    Wall(self, col, row)
                if tile == 'T':
                    Tree(self, col, row)
                if tile == 'P':
                    self.avatar = Avatar(self, col, row)
        self.camera = Camera(self.map.width, self.map.height)
        """

    def new(self):
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        for tile_object in self.map.map_data.objects:
            if tile_object.name == "agent":
                self.agent = Agent(self, tile_object.x, tile_object.y)
            if tile_object.name == "game_border":
                self.game_border = Obstacle(self, tile_object.x, tile_object.y, tile_object.width, tile_object.height)
            if tile_object.name == "water":
                self.water = Obstacle(self, tile_object.x, tile_object.y, tile_object.width, tile_object.height)
            if tile_object.name == "tree_top":
                self.tree_top = Tree_top(self, tile_object.x, tile_object.y, tile_object.width, tile_object.height)
            if tile_object.name == "tree_trunk":
                self.tree_trunk = Tree(self, tile_object.x, tile_object.y, tile_object.width, tile_object.height,
                                           self.tree_top)
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

    def draw(self):
        #self.screen.fill(GREEN)
        #self.draw_grid()
        self.screen.blit(self.map_img, self.camera.apply_rect(self.map_rect))
        for sprite in self.all_sprites:
            self.screen.blit(sprite.image, self.camera.apply(sprite))
        pg.display.flip()

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
        pass

    def show_go_screen(self):
        pass

# create the game object
g = Game()
g.show_start_screen()
while True:
    g.new()
    g.run()
    g.show_go_screen()