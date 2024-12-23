from model import *
import numpy as np

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
                    theta_x = 0
                    theta_y = np.arccos(gr.grid[i,j,k,2])
                    theta_z = np.arctan2(gr.grid[i,j,k,1],gr.grid[i,j,k,0])
                    self.add_object(Vec(app, pos=(i*s, -s*k, j*s),rot=(theta_x,theta_y,theta_z),idx=(i,j,k),scale=(10,10,10)))
                    #self.add_object(Cat(app, pos=(i*s, -s*k, j*s),rot=(theta_x,theta_y,theta_z),idx=(i,j,k),scale=(1,1,1)))

        self.add_object(Cat(app, pos=(0, 0, 0),scale=(0.01,0.01,10))) # marks the z-axis
        #self.add_object(Vec(app, pos=(1, 1, 1),scale=(10,10,10)))

    def render(self,gr=None):
        for obj in self.objects:
            obj.render(gr)