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
        self.asr_folder = path.join(game_folder, "asr")
        self.map = TiledMap(path.join(self.map_folder, "tiled_map.tmx"))
        self.map_img = self.map.make_map()
        self.map_rect = self.map_img.get_rect()
        self.title_font = path.join(self.img_folder, 'arial.ttf')
        self.morgan_speech = SpeechToText(path.join(self.asr_folder, "audio.wav"))


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
        for tile_object in self.map.map_data.objects:
            if tile_object.name == "tree_trunk":
                self.tree_trunk = Tree(self, tile_object.x, tile_object.y, tile_object.width, tile_object.height,
                                       self.tree_top)
        for tile_object in self.map.map_data.objects:
            if tile_object.name == "agent":
                self.agent = Agent(self, tile_object.x, tile_object.y)
        self.camera = Camera(self.map.width, self.map.height)
        self.caption = pg.Rect(0, HEIGHT * 0.72, WIDTH, 40)
        task_goals = [self.tree_top.rect]
        self.tasks = Tasks(task_list, task_goals)

    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def quit(self):
        self.screen.fill(DARKGREEN)
        self.screen.blit(pg.image.load(path.join(self.img_folder, "apple_64px.png")), (WIDTH / 2 - 25, 210))
        pg.display.flip()
        pg.time.delay(200)
        self.screen.blit(pg.image.load(path.join(self.img_folder, "apple_64px_wink.png")), (WIDTH / 2 - 25, 210))
        pg.display.flip()
        pg.time.delay(200)
        self.screen.blit(pg.image.load(path.join(self.img_folder, "apple_64px.png")), (WIDTH / 2 - 25, 210))
        pg.display.flip()
        pg.time.delay(50)
        pg.quit()
        sys.exit()

    def update(self):
        # update portion of the game loop
        self.all_sprites.update()
        self.camera.update(self.agent)
        if self.tasks.task_list:
            self.tasks.check_goal_state(self.agent.rect)


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
        self.display_tasks()
        self.agent.give_text_feedback()
        pg.display.flip()

    def display_tasks(self):
        textRect = pg.Rect(0, 0, 0, 0)
        font = pg.font.Font(self.title_font, 15)
        height = 0
        for task in self.tasks.task_list:
            textSurf = font.render(task, True, BLACK).convert_alpha()
            textSize = textSurf.get_size()
            height += textSize[0]
            bubbleSurf = pg.Surface((textSize[0] * 2., textSize[1] * 2))
            textRect = bubbleSurf.get_rect()
            bubbleSurf.fill(LIGHTGREY)
            bubbleSurf.blit(textSurf, textSurf.get_rect(center=textRect.center))
            textRect.center = ((700), (height))
            self.screen.blit(bubbleSurf, textRect)

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
        self.draw_text("Hello, and welcome to the world of me, Young Apple.", self.title_font, 30, WHITE, WIDTH / 2,
                       HEIGHT / 3, align="center")
        self.screen.blit(pg.image.load(path.join(self.img_folder, "apple_64px.png")), (WIDTH / 2 - 25, 210))
        self.draw_text("I'm ready to move around and learn new tricks.", self.title_font, 30, WHITE, WIDTH / 2,
                       HEIGHT * 2 / 3, align="center")
        self.draw_text("Press any key to start.", self.title_font, 20, LIGHTLIGHTGREY,
                       WIDTH / 2, HEIGHT * 13 / 16, align="center")
        self.draw_text("Press 'escape' to exit.", self.title_font, 15, LIGHTLIGHTGREY,
                       WIDTH / 2, HEIGHT * 14 / 16, align="center")
        pg.display.flip()
        self.wait_for_key()


#     def name_agent_screen(self): # with text
#         # Give Young Apple a name
#         # name = g.name_agent()
#         # g.agent.give_name(name.lower())
#         confirm = "n"
#         while confirm.lower()=="n":
#             name = input("\nWhat would you like to call me when teaching me tricks? ")
#             confirm = input("Call me, '" + name + "'? (y/n) ")
#         print("Terrific. '" + name +"' is my name!" )

#         self.agent.give_name(name.lower())
#         print("Teach me, " + name + ", to: climb the tree.")

    def name_agent_screen(self):
        name = ""
        confirm = False
        while not confirm:
            #print("start test")
            self.screen.fill(DARKGREEN)
            self.draw_text("What would you like to call me when teaching me tricks?", self.title_font, 30, WHITE, WIDTH / 2,
                           HEIGHT / 3, align="center")
            self.screen.blit(pg.image.load(path.join(self.img_folder, "apple_64px.png")), (WIDTH / 2 - 25, 210))
            self.draw_text("I can listen to you while you press 'SPACE' or 'm'!", self.title_font, 30, WHITE, WIDTH / 2,
                           HEIGHT * 2 / 3, align="center")
            pg.display.flip()
            pg.event.wait()
            keys = pg.key.get_pressed()
            if keys[pg.K_SPACE]:
                self.draw_text("Listening...", self.title_font, 20, WHITE, WIDTH / 2, HEIGHT * 3 / 4, align="center")
                pg.display.flip()
                #print("listening...")
                with sr.Microphone() as source:
                    try:
                        audio = r.listen(source, timeout=5)
                        name = r.recognize_google(audio)
                        #print("name assigned")
                        self.screen.fill(DARKGREEN, rect=self.caption)
                        pg.display.flip()
                    except:
                        #print("I did not hear anything")
                        self.screen.fill(DARKGREEN, rect=self.caption)
                        pg.display.flip()
                        self.draw_text("Hm? Can you please say that again?", self.title_font, 20, WHITE, WIDTH / 2,
                                   HEIGHT * 3 / 4, align="center")
                        pg.display.flip()
                        pg.time.delay(2000)
            elif keys[pg.K_m]:
            	#print("listening...")
            	with sr.Microphone() as source:
                    try:
                        audio = r.listen(source, timeout=5)
                        #print("audio")
                        self.morgan_speech.saveAudio(audio)
                        name = self.morgan_speech.getTranscription()
                        self.screen.fill(DARKGREEN, rect=self.caption)
                        pg.display.flip()
                        #print("name assigned")
                    except:
                        #print("I did not hear anything")
                        self.screen.fill(DARKGREEN, rect=self.caption)
                        pg.display.flip()
                        self.draw_text("Hm? Can you please say that again?", self.title_font, 20, WHITE, WIDTH / 2,
                                       HEIGHT * 3 / 4, align="center")
                        pg.display.flip()
                        pg.time.delay(2000)
            elif keys[pg.K_ESCAPE]:
                self.quit()
            while name and not confirm:
                #print("confirmation step")
                self.draw_text("Do you want to call me "+name+"? ENTER/n", self.title_font, 20, WHITE, WIDTH / 2, HEIGHT * 3 / 4, align="center")
                pg.display.flip()
                pg.event.wait()
                self.clock.tick(FPS)
                keys = pg.key.get_pressed()
                if keys[pg.K_RETURN]:
                    confirm = True
                    self.screen.fill(DARKGREEN)
                    self.draw_text("Terrific. '" + name + "' is my name!", self.title_font, 20, WHITE,
                                   WIDTH / 2, HEIGHT * 3 / 4, align="center")
                    pg.display.flip()
                    pg.time.delay(2000)
                elif keys[pg.K_n]:
                    name = ""
        return name.lower()


    def show_go_screen(self):
        pass

    def name_agent(self):
        '''
        User names the game agent. Returns string name of agent. 
        '''
        name = input("\nWhat would you like to call me when teaching me tricks? ")
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
g.new()
g.show_start_screen()

name = g.name_agent_screen()

while True:
    #g.new()
    g.agent.give_name(name)

    g.run()
    g.show_go_screen()