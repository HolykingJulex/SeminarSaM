#Parameters for rhombo hematite
import numpy as np
from scipy import constants

UNIT_CELL_X1		=	1 
UNIT_CELL_X2		=	1
UNIT_CELL_X3		=	1

UNIT_CELL_DIM_X1 =	1
UNIT_CELL_DIM_X2 =	1
UNIT_CELL_DIM_X3 =	1

def putStructure(i,j,k):

	return 1

def getPosition(i,j,k):
      
	return [ (UNIT_CELL_X1 / UNIT_CELL_DIM_X1) * i,
             (UNIT_CELL_X2 / UNIT_CELL_DIM_X2) * j,
             (UNIT_CELL_X3 / UNIT_CELL_DIM_X3) * k ]

def getdistancevec(i,j,k, ti,tj,tk):
    r1 = np.array(getPosition(i,j,k))
    r2 = np.array(getPosition(ti,tj,tk))

    return r1-r2









