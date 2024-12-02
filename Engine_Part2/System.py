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
   

def normalize(temp) -> ndarray:
        norm = np.linalg.norm(temp)
        if norm == 0: 
            raise ValueError('Some of your vectors have a norm of 0. I cant normalise that')
        return temp / norm
    
def getnormspin():

    temp = np.array([rd.random(),rd.random(),rd.random()])
    while np.linalg.norm(temp) == 0:  #making sure we dont acidentally get [0,0,0]
            temp = np.array([rd.random(),rd.random(),rd.random()])
    #return normalize(temp)
    return  normalize(np.array([0,0,1]))


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
    #field:ndarray[(Any,Any,Any,3),float]
    def __init__(self,size,vec) -> None:

        self.field = np.ndarray((size[0],size[1],size[2],3),float)
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
    #grid:ndarray[(Any, Any,Any,3), float]
    #temperature_Field:TempField
    mag_Field:MagField 
    onSiteAnisotropies:OnSiteAnisotropies
    startTime = 0
    endTime = 10
    timestep = 10**(-14)
    #timestep = 0.01
    currentTime = 0
    matshells:Matshell
    alpha:float
    gamma:float
    mu:float

    def __init__(self,size,shellfolder,*args) -> None:
        #TODO clever inputs here
        self.matshells = get_Matshell(shellfolder) 
        self.size = size
        self.grid = self.init_grid()
        self.mag_Field = MagField(self.size,np.array([0,0,0])) 
        #self.temperature_Field = TempField(self.size,0)
        self.onSiteAnisotropies = OnSiteAnisotropies(self.size,0)
        self.alpha = self.matshells[0].alpha
        self.gamma = self.matshells[0].gamma
        self.mu = self.matshells[0].mu
        pass
    
    def init_grid(self):
        data = np.ndarray((self.size[0],self.size[1],self.size[2],3),float)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    data[i,j,k] = getnormspin()   # This constructor creates empty spins
        return data
    
    def returnangles(self,i,j,k):
        dir = self.grid[i,j,k]
        xan = 0
        yan = 0
        zan = 0
        return (xan,yan,zan)
    
    def rotatingBfield(self):
        #field = np.ndarray((self.size[0],self.size[1],self.size[2],3),float)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    #if j > self.size[1]-1:
                    if j < 1:
                        #self.grid[i,j,k] =  normalize( np.array((np.sin(self.currentTime),np.cos(self.currentTime),5)))
                        self.grid[i,j,k] =  normalize( np.array((np.sin(self.currentTime*3.5*10**12),np.cos(self.currentTime*3.5*10**12),3)))
                        #if i <1:
                        #    print(np.sin(self.currentTime*10**13))
                        #    print(i,j,k)
                        #    print(self.grid[i,j,k])
                        #temp = np.array([rd.random(),rd.random(),rd.random()])
                        #while np.linalg.norm(temp) == 0:  #making sure we dont acidentally get [0,0,0]
                        #        temp = np.array([rd.random(),rd.random(),rd.random()])
                        #self.grid[i,j,k] = normalize(temp)
                        #self.mag_Field.field[i,j,k] =  normalize( np.array((np.sin(self.currentTime),np.cos(self.currentTime),100)))
                        #self.mag_Field.field[i,j,k] =   np.array((0,0,1))
                    
        return self.mag_Field.field

    def update(self):
        self.currentTime = self.currentTime + self.timestep
        
        Heff = self.calculateExchangeVectors(self.grid)                  #This should be a vector
        #Heff += self.mag_Field.field * 10**(-12)                         #For B-Field this is also a vector
        Heff += self.rotatingBfield()                         #For B-Field this is also a vector
        Heff += self.calculateAnisVectors(self.grid)                         
                                                        

        # Now we do the guess step in Heuns method

        grid_tilde = self.grid + self.timestep * self.LLG(self.grid,Heff)

        # Now we do the corrector step in Heuns method

        Heff = self.calculateExchangeVectors(grid_tilde)                    #This should be a vector
        Heff += self.mag_Field.field                                        #For B-Field this is also a vector
        Heff += self.calculateAnisVectors(grid_tilde) 

        new_grid = self.grid + 0.5 * self.timestep * (grid_tilde + self.LLG(grid_tilde,Heff) )

        #Need to normalie this beforehand
        new_grid = self.normalise(new_grid)                                                     #TODO make faster

        self.grid = new_grid

        pass
    
    def normalise(self,grid):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    vec = grid[i,j,k]
                    grid[i,j,k] = vec/np.linalg.norm(vec)
        return grid

    def __str__(self):
        return np.array_repr(self.grid.data)
    
    def __repr__(self):
        return np.array_repr(self.grid.data)
    
    def LLG(self,grid,Heff):
        firstterm = np.cross(grid,Heff)
        secondterm = self.alpha * np.cross(grid,firstterm)
        return - self.gamma/((1+self.alpha**2)*self.mu) * (firstterm+secondterm)


    def calculateExchangeVectors(self,grid):
        field = np.zeros((self.size[0],self.size[1],self.size[2],3), float)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    
                    sl = putStructure(i,j,k) 
                    matshell = self.matshells[sl-1]
                    vec = np.zeros((1,3),float)            #Todo check if this causes wierd behaviour
                    
                    for Jtens in matshell.Js:               #Going through all the interaction neigbors
                        oi = (i+Jtens.i) % self.size[0]
                        #oj = (j+Jtens.j) % self.size[1]     #wo dont want periodic in this direction
                        oj = j+Jtens.j     
                        ok = (k+Jtens.k) % self.size[2]
                        if (oj < self.size[1]) & (oj >-1):
                            vec += (Jtens.matrix @ grid[oi,oj,ok])
                                                
                    field[i,j,k] = vec

        return field

    def calculateAnisVectors(self,grid):
        field = np.zeros((self.size[0],self.size[1],self.size[2],3), float)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):                    
                    
                    sl = putStructure(i,j,k) 
                    matshell = self.matshells[sl-1]
                    vec = (matshell.d2 @ grid[i,j,k])
                    field[i,j,k] = vec

        return field

    


