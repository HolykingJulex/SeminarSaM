import numpy as np
from numpy import ndarray
from typing import Any
from typing import List
import random as rd
import os
import re
from scipy import constants
from copy import deepcopy

from structure.sc_FM import *

class Constants:
    def __init__(self):
        self.h       = 6.62607015e-34
        self.hbar    = self.h/(2*np.pi)
        self.k_B     = 1.380649e-23
        self.e       = 1.602176634e-19
        self.m_e     = 9.1093837015e-31
        self.mu_B    = self.e*self.hbar/(2*self.m_e)
        self.gamma_e = 1.76085963023e11
        self.mu_0    = 1.25663706212e-6
        self.c       = 299792458
        self.Ry      = 2.1798723611035e-18
        self.meV     = constants.milli * constants.electron_volt


cons = Constants()

class InteractionMatrix:
    def __init__(self,index,Jxx,Jxy,Jxz,Jyx,Jyy,Jyz,Jzx,Jzy,Jzz):
        self.index = index
        self.i =index[0]
        self.j =index[1]
        self.k =index[2]
        self.matrix = np.matrix([[Jxx,Jxy,Jxz],[Jyx,Jyy,Jyz],[Jzx,Jzy,Jzz]])

    def __str__(self) -> str:
        return str(self.index) + str( self.matrix)
    
class Matshell:
    def __init__(self,*args, **kwargs):
        self.sl = kwargs.get("slindex")
        self.slindex = self.sl
        self.slname = kwargs.get("slname")
        self.Sz = kwargs.get("Sz")
        self.gamma = kwargs.get("gamma")
        self.alpha = kwargs.get("alpha")
        self.mu = kwargs.get("mu")
        self.d2 = kwargs.get("d2")
        self.d4 = kwargs.get("d4")
        self.d6 = kwargs.get("d6")
        self.d66 = kwargs.get("d66")
        self.Js = kwargs.get("Js")
    def __str__(self) -> str:
        return str(self.sl) +" "+ str( self.slname)+" "+ str( self.gamma)+" "+ str( self.Sz)+" " + str( self.alpha)+" "+ str( self.mu)+" "+ str( self.d2)+ str( self.d4)+" "+ str( len(self.Js))+"\n"
    def __repr__(self):
        return str(self)
    
  


class Spin:
    vec = np.array([0,0,0])

    #just three dimensional numpy vector for now
    def __init__(self) -> None:
        temp = np.array([rd.random(),rd.random(),rd.random()])
        
        while np.linalg.norm(temp) == 0:  #making sure we dont acidentally get [0,0,0]
            temp = np.array([rd.random(),rd.random(),rd.random()])

        self.vec = self.normalize(temp)
        pass
    
    def __mul__(self,other):
        return self.vec @ other.vec

    def __rmul__(self,other):
        return self.vec @ other.vec

    def normalize(self,temp) -> ndarray:
        norm = np.linalg.norm(temp)
        if norm == 0: 
            raise ValueError('Some of your vectors have a norm of 0. I cant normalise that')
        return temp / norm

    def __str__(self)-> str:
        return self.vec
    
    def __repr__(self)-> str:
        #print(type( self.vec.tostring()))
        return np.array_repr(self.vec)



class TempField:
    #contains either scalar or vectorial value for each spin
    field:ndarray[(Any,Any),float]
    def __init__(self,size,temp) -> None:
        
        self.field = np.ndarray(size,object)
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    self.field[i,j,k] = temp   
        pass

class OnSiteAnisotropies:
    #contains either scalar or vectorial value for each spin
    tensors:None
    def __init__(self,size,tensor) -> None:
        
        self.tensors = np.ndarray(size,object)
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    self.tensors[i,j,k] = tensor   
       
        pass


class MagField:
    #contains either scalar or vectorial value for each spin
    field:ndarray[(Any,Any),ndarray]
    def __init__(self,size,vec) -> None:

        self.field = np.ndarray(size,object)
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    self.field[i,j,k] = vec   

        pass
    
    def update(self,time)-> None:
        print("need to implemnt time dependence of the magnetic field")
    
    
def get_Matshell(matfolder)-> List:
     Matshells = []
     counter = 0
     breakpoint = 1000

     for name in sorted(os.listdir(matfolder)):
          if os.path.isfile(os.path.join(matfolder, name)):
               counter += 1
               with open(os.path.join(matfolder, name)) as mat_file:
                    Js = []
                    for index,line in enumerate(mat_file):

                         if (index == 0)|(index == 2):
                              if not line.startswith("#"):
                                   if line.startswith("F"):
                                        Sz = 1
                                        if (counter == 2) | (counter == 3):#carefull this only works for rhombo hematite or simple Fm Afm
                                             Sz = -1
                                        if (counter == 6) | (counter == 7):#carefull this only works for rhombo hematite or simple Fm Afm
                                             Sz = -1
                                        current_mat = Matshell(slname=line.strip("\n"),slindex = counter,Sz = Sz)
                                        print(line.strip("\n"), counter, Sz, "from file " ,name)

                         if line.startswith("gamma"):
                              current_mat.gamma = float(re.findall("\d+\.\d+", line)[0]) * cons.gamma_e# need gyromagnetic ratio here in the right units
                         
                         if line.startswith("alpha"):
                              #current_mat.alpha = float(re.findall("\d+\.\d+", line)[0]) #problem with scientific notation 
                              current_mat.alpha = 0.00001

                         if line.startswith("mu"):
                              current_mat.mu = float(re.findall("\d+\.\d+", line)[0]) * cons.mu_B

                         if line.startswith("d2"):
                              a = np.zeros([3, 3])
                              np.fill_diagonal(a, re.findall("\d+\.\d+", line))
                              current_mat.d2 = a * cons.meV 

                         if line.startswith("d4"):
                              a = np.zeros((3, 3, 3, 3))
                              val = re.findall("\d+\.\d+", line)[-1]
                              a[2,2,2,2]=float(val)
                              current_mat.d4 = a * cons.meV

                         if line.startswith("d6"):
                              a = np.zeros((3, 3))
                              np.fill_diagonal(a, re.findall("\d+\.\d+", line))
                              current_mat.d6 = a 

                         if line.startswith("d66"):
                              current_mat.d6 = re.findall("\d+\.\d+", line)[0]

                         if line.startswith("#") & (index >0):
                              breakpoint = index

                         if index > breakpoint:
                              v = line.split()                         
                              if v:                              #checking if non empty
                                   #print(v)
                                   J = InteractionMatrix([int(v[0]),int(v[1]),int(v[2])],
                                                       float(v[3])* cons.meV,float(v[4])* cons.meV ,float(v[5])* cons.meV,
                                                       float(v[6])* cons.meV,float(v[7])* cons.meV ,float(v[8])* cons.meV ,
                                                       float(v[9])* cons.meV,float(v[10])* cons.meV,float(v[11])* cons.meV)
    
                                   Js.append(J)
                    #Js  = Js[-6:]
                    #Js  = Js[:-6]
                    current_mat.Js = Js
                    #print(current_mat.Js)
                    Matshells.append(current_mat)

              
     return Matshells
       


class System:
    size:None
    grid:ndarray[(Any, Any), Spin]
    temperature_Field:TempField
    mag_Field:MagField 
    onSiteAnisotropies:OnSiteAnisotropies
    startTime = 0
    endTime = 10
    timestep = 0.001
    currentTime = 0
    matshells:Matshell

    def __init__(self,size,shellfolder,*args) -> None:
        #TODO clever inputs here
        self.matshells = get_Matshell(shellfolder) 
        self.size = size
        self.grid = self.init_grid()
        self.mag_Field = MagField(self.size,np.array([0,0,0])) # how do we update this
        self.temperature_Field = TempField(self.size,0)
        self.onSiteAnisotropies = OnSiteAnisotropies(self.size,0)
        pass
    
    def init_grid(self):
        data = np.ndarray(self.size,Spin)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    data[i,j,k] = Spin()   # This constructor creates empty spins
        return data

    def update(self):
        self.currentTime = self.currentTime + self.timestep
        oldGrid = deepcopy(self.grid) #maybe not needed but good as safty
        #update field first. No time in temp needed for now
        test = self.calculateExchange(oldGrid)
        #get effective field
        #Hef = foo(oldGrid)
        
        #make new field based on old field and the effective field and temps
        #self.Grid = foo(oldGrid,Hef,temp)
        return test


    def __str__(self):
        return np.array_repr(self.grid.data)
    
    def __repr__(self):
        return np.array_repr(self.grid.data)

    def calculateExchange(self,grid)-> float:
        out = 0
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    print("got shell here")
                    print(i,j,k)
                    sl = putStructure(i,j,k) 
                    matshell = self.matshells[sl-1]
                    
                    currentspin = self.grid[i,j,k]
                    
                    for Jtens in matshell.Js:               #Going through all the interaction neigbors
                        oi = (i+Jtens.i) % self.size[0]
                        oj = (j+Jtens.j) % self.size[1]
                        ok = (k+Jtens.k) % self.size[2]
                        dotpr = currentspin * self.grid[oi,oj,ok]
                        print(Jtens.j,[oi,oj,ok])


        return 0
    
    def exchange_interaction_field(self,oldgrid):
        out = np.zeros(shape=(self.grid.shape))
        return oldgrid

    def anisotropy_interaction_field(self):
        return self.grid
    
    def dS_llg(self,grid,Heff):
        return self.grid
    
    def normalise(self):
        for index,sp in enumerate(self.grid):
            self.grid[index] = sp.normalise()
    
    def integrate(self) -> float:
    # compute external fields. These fields does not change
    # because they don't depend on the state
        oldGrid = deepcopy(self.grid)
        #Hext = self.temperature_Field.field 'TODO figure out how to do this
        Hext = self.mag_Field.field

        # predictor step

        # compute the effective field as the sum of external fields and
        # spin fields
        Heff = Hext + self.exchange_interaction_field(oldGrid)
        Heff = Heff + self.anisotropy_interaction_field(oldGrid)

        # compute dS based on the LLG equation
        dS = self.dS_llg(self.grid,Heff)

        # compute the state_prime
        state_prime = self.grid + self.timestep * dS

        # normalize state_prime
        state_prime = self.normalize(state_prime)

        # corrector step

        # compute the effective field prime by using the state_prime. We
        # use the Heff variable for this in order to reutilize the memory.
        Heff = Hext + self.exchange_interaction_field( )
        Heff = Heff + self.anisotropy_interaction_field()

        # compute dS_prime employing the Heff prime and the state_prime
        dS_prime = self.dS_llg(state_prime, Heff)

        # compute the new state
        integrate = self.grid + 0.5 * (dS + dS_prime) * self.timestep

        # normalize the new state
        return self.normalize(integrate)

