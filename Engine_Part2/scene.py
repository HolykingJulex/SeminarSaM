from model import *


class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load(app.gr)

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self,gr):
        app = self.app
       
        n1,n2,n3, s = gr.size[0],gr.size[1],gr.size[2], 3
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    self.add_object(Cube(app, pos=(i*s, -s*k, j*s),rot=(gr.grid[i,j,k,0],gr.grid[i,j,k,1],gr.grid[i,j,k,2])))

        #self.add_object(Cat(app, pos=(0, -2, -10)))

    def render(self):
        for obj in self.objects:
            obj.render()