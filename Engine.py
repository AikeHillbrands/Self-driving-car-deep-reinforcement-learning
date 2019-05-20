import numpy as np
import pygame as pg
import math

class Engine:
    dispSize = (1600,900)
    frameRate =40
    scale = 50
    def __init__(self):
        pg.init()
        pg.display.init()
        self.screen = pg.display.set_mode(self.dispSize, pg.SWSURFACE | pg.DOUBLEBUF)

        self.car_img = pg.transform.rotozoom(pg.image.load("car.png"),-90,self.scale/200)

    def new_frame(self):
        self.screen.fill((255, 255, 255))
    
    def draw_lines(self,points):
        pg.draw.lines(self.screen,(0,0,0),False,np.multiply(points,self.scale),2)
    
    def draw_car(self,pos,car_dir):
        car = pg.transform.rotozoom(self.car_img,math.degrees(math.atan2(-car_dir[1],car_dir[0])),1)
        self.screen.blit(car,(pos[0]*self.scale-car.get_size()[0]/2,pos[1]*self.scale-car.get_size()[1]/2))

    def render(self):
        pg.display.flip()

    def close(self):
        pg.quit()

    def wait(self,time):
        pg.time.wait(int(time))