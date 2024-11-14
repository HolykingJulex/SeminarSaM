from model import *


class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load()

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
       
        n, s = 30, 3
        for x in range(-n, n, s):
            for z in range(-n, n, s):
                self.add_object(Cube(app, pos=(x*1.5, -s, z*1.5)))

        #self.add_object(Cat(app, pos=(0, -2, -10)))

    def render(self):
        for obj in self.objects:
            obj.render()