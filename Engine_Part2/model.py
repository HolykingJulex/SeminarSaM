import moderngl as mgl
import numpy as np
import glm


class BaseModel:
    def __init__(self, app, vao_name, tex_id, pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        self.app = app
        self.pos = pos
        self.rot = glm.vec3([glm.radians(a) for a in rot])
        self.scale = scale
        self.m_model = self.get_model_matrix()
        self.tex_id = tex_id
        self.vao = app.mesh.vao.vaos[vao_name]
        self.program = self.vao.program
        self.camera = self.app.camera

    def update(self): ...

    def get_model_matrix(self):
        m_model = glm.mat4()
        # translate
        m_model = glm.translate(m_model, self.pos)
        # rotate
        
        m_model = glm.rotate(m_model, self.rot.y, glm.vec3(0, 1, 0))
        m_model = glm.rotate(m_model, self.rot.x, glm.vec3(1, 0, 0))
        m_model = glm.rotate(m_model, self.rot.z, glm.vec3(0, 0, 1))
        # scale
        m_model = glm.scale(m_model, self.scale)
        return m_model

    def render(self,gr=None):
        self.update(gr)
        self.vao.render()


class Cube(BaseModel):
    def __init__(self, app, vao_name='cube', tex_id=0, pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1),idx = (0,0,0)):
        super().__init__(app, vao_name, tex_id, pos, rot, scale)
        self.on_init()
        self.idx = idx

    def update(self,gr):
        self.texture.use()

        #rotate to 0,0,1
        self.m_model = glm.rotate(self.m_model, -self.rot.z, glm.vec3(0, 0, 1))   #this make the tilting
        self.m_model = glm.rotate(self.m_model, -self.rot.y, glm.vec3(0, 1, 0))   #this should make a rotation around the z-axis 
        

        # Calculating the angles
        theta_x = 0
        theta_z = np.arccos(gr.grid[self.idx[0],self.idx[1],self.idx[2],2])
        theta_y = np.arctan2(gr.grid[self.idx[0],self.idx[1],self.idx[2],1],gr.grid[self.idx[0],self.idx[1],self.idx[2],0])
        #print("theta_z ",theta_z)
        
        # Putting the difference compared to the last angles in a vec
        self.rot = glm.vec3([theta_x-self.rot.x,
                             theta_y-self.rot.y,
                             theta_z-self.rot.z])
        self.rot = glm.vec3([theta_x,theta_y,theta_z])
        #print("it should face in this direction ",gr.grid[self.idx[0],self.idx[1],self.idx[2]])
        #

        #TODO find out the coordinate system, i somehow assume that z is outofMonitor 
        
        # Rotating by the difference of the angles
        self.m_model = glm.rotate(self.m_model, self.rot.y, glm.vec3(0, 1, 0))   #this should make a rotation around the z-axis 
        self.m_model = glm.rotate(self.m_model, self.rot.z, glm.vec3(0, 0, 1))   #this make the tilting
        
        #self.m_model = glm.rotate(self.m_model, self.rot.x, glm.vec3(1, 0, 0)) #shouldnt do anything
     


        #PROBLEM the scales are fixed in x,y,z dir NOT body dir used scales object insteatd

        #anglenorm  = (theta_x+theta_y+theta_z)/3
        #scales = (theta_x/anglenorm+1, theta_y/anglenorm+1, theta_z/anglenorm+1)
        #scales = (1,1,1)

        #self.scale = (1,1/self.scale[1],1/self.scale[2]) #Redoing the last scale
        #self.m_model = glm.scale(self.m_model, self.scale) # need to update scale aswell ugh


        #self.scale = (1,2,1)                            # Making the new scale 
        #self.scale = scales                            # Making the new scale 
        #self.m_model = glm.scale(self.m_model, self.scale) # need to update scale aswell ugh

        #Setting the current rotation 
        self.rot = glm.vec3([theta_x,theta_y,theta_z])
        #print("but is rotated in this direction as angle ",np.rad2deg(self.rot))

        self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def on_init(self):
        # texture
        self.texture = self.app.mesh.texture.textures[self.tex_id]
        self.program['u_texture_0'] = 0
        self.texture.use()
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        self.program['light.Is'].write(self.app.light.Is)


class Cat(BaseModel):
    def __init__(self, app, vao_name='cat', tex_id='cat',
                 pos=(0, 0, 0), rot=(-90, 0, 0), scale=(1, 1, 1),idx=(0,0,0)):
        super().__init__(app, vao_name, tex_id, pos, rot, scale)
        self.on_init()

    def update(self,gr):
        self.texture.use()
        self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def on_init(self):
        # texture
        self.texture = self.app.mesh.texture.textures[self.tex_id]
        self.program['u_texture_0'] = 0
        self.texture.use()
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        self.program['light.Is'].write(self.app.light.Is)

