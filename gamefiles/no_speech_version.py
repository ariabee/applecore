import pygame as pg
import sys
from settings import *
from sprites import *
from map import *
import pathlib


class Game:

    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        self.clock = pg.time.Clock()
        self.load_data()

    def load_data(self):
        game_folder = pathlib.Path(__file__).parent
        img_folder = pathlib.Path(game_folder / 'img')
        self.map_folder = pathlib.Path(game_folder / 'maps')
        self.dim_screen = pg.Surface(self.screen.get_size()).convert_alpha()
        self.dim_screen.fill((0, 0, 0, 180))
        self.avatar_img = pg.image.load(str(pathlib.Path(img_folder / 'avatar.png'))).convert_alpha()

    def new(self):
        # initialize all variables and do all the setup for a new game
        #self.all_sprites = pg.sprite.Group()
        self.all_sprites = pg.sprite.LayeredUpdates()
        self.walls = pg.sprite.Group()
        self.map = TiledMap(str(pathlib.Path(self.map_folder / 'level1.txt')))
        self.map_img = self.map.make_map()
        self.map.rect = self.map_img.get_rect()
        for tile_object in self.map.tmxdata.objects:
            obj_center = vec(tile_object.x + tile_object.width / 2,
                             tile_object.y + tile_object.height / 2)
            if tile_object.name == 'player':
                self.avatar = Avatar(self, obj_center.x, obj_center.y)
            if tile_object.name == 'wall':
                Obstacle(self, tile_object.x, tile_object.y,
                         tile_object.width, tile_object.height)
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
        self.camera.update(self.avatar)

    def draw_grid(self):
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw(self):
        self.screen.blit(self.map_img, self.camera.apply(self.map))
        for sprite in self.all_sprites:
            self.screen.blit(sprite.image, self.camera.apply(sprite))
        pg.display.flip()

    def events(self):
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()
                if event.key == pg.K_LEFT:
                    self.avatar.move(dx=-1)
                if event.key == pg.K_RIGHT:
                    self.avatar.move(dx=1)
                if event.key == pg.K_UP:
                    self.avatar.move(dy=-1)
                if event.key == pg.K_DOWN:
                    self.avatar.move(dy=1)

    def show_start_screen(self):
        pass

    def show_go_screen(self):
        pass

g = Game()
g.show_start_screen()
while True:
    g.new()
    g.run()
    g.show_go_screen()


"""
# Setting up Fonts
font = pg.font.SysFont("Verdana", 60)
font_small = pg.font.SysFont("Verdana", 20)
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
"""