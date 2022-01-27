import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy import optimize
import math
from matplotlib.patches import Polygon


import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy import optimize
import math
from matplotlib.patches import Polygon

class CellBox:
  def __init__(self,X,Y,dx,dy,value=None,isLand=None,vector=np.array([0.,0.])):
    # Information about the Cell Block
    self.x  = X;  self.y = Y; 
    self.cx = self.x + dx/2.; self.cy = self.y + dy/2.
    self.dy = dy; self.dx = dx
    self.dyp=dy/2;self.dxp=dx/2;self.dym=dy/2;self.dxm=dx/2

    # These include the corner points and the half-way points in the edge 
    self._bounding_points = [[self.x,self.y],
                             [self.x,self.y+self.dy/2],
                             [self.x,self.y+self.dy],
                             [self.x+self.dx/2,self.y+self.dy],
                             [self.x+self.dx,self.y+self.dy],
                             [self.x+self.dx,self.y+self.dy/2],
                             [self.x+self.dx,self.y],
                             [self.x+self.dx/2,self.y]]
    
    # Cell Parameters
    self.value          = value
    self.isLand         = isLand
    self.vector         = vector

    # Splitting Information
    self._leafDepth     = 0
      
  def _splitDomain(self):
    return np.array([[self.x,self.y],[self.x,self.cy],[self.cx,self.cy],[self.cx,self.y]]), self.dx/2, self.dy/2 

  def _define_waypoints(self,pt):
    self.cx,self.cy = pt
    self.dxp = (self.x + self.dx) - self.cx
    self.dxm = self.cx - self.x
    self.dyp = (self.y + self.dy) - self.cy
    self.dym = self.cy - self.y