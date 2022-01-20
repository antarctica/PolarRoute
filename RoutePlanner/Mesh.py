import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy import optimize
import math
from matplotlib.patches import Polygon

from RoutePlanner.CellBox import CellBox

def f(y,σ_a,σ_b,ShipSpeed,dim):
    if dim == 1:
      x = distance((σ_a.cx,σ_a.cy),(σ_a.cx,σ_a.cy+np.sign(σ_b.cy-σ_a.cy)*(σ_a.dy/2)))
      y = distance((σ_a.cx,σ_a.cy+np.sign(σ_b.cy-σ_a.cy)*(σ_a.dy/2)),(σ_a.cx+y,σ_a.cy+np.sign(σ_b.cy-σ_a.cy)*(σ_a.dy/2)))
      a = distance((σ_b.cx,σ_b.cy),(σ_b.cx,σ_a.cy-np.sign(σ_b.cy-σ_a.cy)*(σ_b.dy/2)))
      Y = distance( (σ_a.cx, σ_a.cy+np.sign(σ_b.cy-σ_a.cy)*((σ_a.dy/2)+(σ_b.dy/2)) ),( σ_b.cx, σ_a.cy+np.sign(σ_b.cy-σ_a.cy)*((σ_a.dy/2)+(σ_b.dy/2)) ))
      ua = σ_a.vector[1]; va = σ_a.vector[0];
      ub = σ_b.vector[1]; vb = σ_b.vector[0];
    if dim == 0:
      x = distance((σ_a.cx,σ_a.cy),(σ_a.cx+np.sign(σ_b.cx-σ_a.cx)*(σ_a.dx/2),σ_a.cy))
      y = distance((σ_a.cx+np.sign(σ_b.cx-σ_a.cx)*(σ_a.dx/2),σ_a.cy),(σ_a.cx+np.sign(σ_b.cx-σ_a.cx)*(σ_a.dx/2),σ_a.cy+y))
      a = distance((σ_b.cx,σ_b.cy),(σ_b.cx-np.sign(σ_b.cx-σ_a.cx)*(σ_b.dx/2),σ_b.cy))
      Y = distance((σ_a.cx + np.sign(σ_b.cx-σ_a.cx)*((σ_a.dx/2)+(σ_b.dx/2)), σ_a.cy),( σ_a.cx + np.sign(σ_b.cx-σ_a.cx)*((σ_a.dx/2)+(σ_b.dx/2)), σ_b.cy))
      ua = σ_a.vector[0]; va = σ_a.vector[1];
      ub = σ_b.vector[0]; vb = σ_b.vector[1];

    # --- Determining Newtonian Minimisation function based on Appendix 1 ---
    d_a = x**2 + y**2; d_b = a**2 + (Y-y)**2
    θ_a = ShipSpeed**2 - ua**2 - va**2; θ_b = ShipSpeed**2 - ub**2 - vb**2   
    D_a = x*ua + y*va; D_b = a*ub + (Y-y)*vb
    X_a = np.sqrt(D_a**2 + θ_a*(d_a**2)); X_b = np.sqrt(D_b**2 + θ_b*(d_b**2))

    F = X_a*( y - Y + (vb*(X_b-D_b))/θ_b) + X_b*(y - (va*(X_a-D_a))/θ_a)
    return F


def T(y,σ_a,σ_b,ShipSpeed,dim):
    if dim == 1:
      x = distance((σ_a.cx,σ_a.cy),(σ_a.cx,σ_a.cy+np.sign(σ_b.cy-σ_a.cy)*(σ_a.dy/2)))
      y = distance((σ_a.cx,σ_a.cy+np.sign(σ_b.cy-σ_a.cy)*(σ_a.dy/2)),(σ_a.cx+y,σ_a.cy+np.sign(σ_b.cy-σ_a.cy)*(σ_a.dy/2)))
      a = distance((σ_b.cx,σ_b.cy),(σ_b.cx,σ_a.cy-np.sign(σ_b.cy-σ_a.cy)*(σ_b.dy/2)))
      Y = distance( (σ_a.cx, σ_a.cy+np.sign(σ_b.cy-σ_a.cy)*((σ_a.dy/2)+(σ_b.dy/2)) ),( σ_b.cx, σ_a.cy+np.sign(σ_b.cy-σ_a.cy)*((σ_a.dy/2)+(σ_b.dy/2)) ))
      ua = σ_a.vector[1]; va = σ_a.vector[0];
      ub = σ_b.vector[1]; vb = σ_b.vector[0];
    if dim == 0:
      x = distance((σ_a.cx,σ_a.cy),(σ_a.cx+np.sign(σ_b.cx-σ_a.cx)*(σ_a.dx/2),σ_a.cy))
      y = distance((σ_a.cx+np.sign(σ_b.cx-σ_a.cx)*(σ_a.dx/2),σ_a.cy),(σ_a.cx+np.sign(σ_b.cx-σ_a.cx)*(σ_a.dx/2),σ_a.cy+y))
      a = distance((σ_b.cx,σ_b.cy),(σ_b.cx-np.sign(σ_b.cx-σ_a.cx)*(σ_b.dx/2),σ_b.cy))
      Y = distance((σ_a.cx + np.sign(σ_b.cx-σ_a.cx)*((σ_a.dx/2)+(σ_b.dx/2)), σ_a.cy),( σ_a.cx + np.sign(σ_b.cx-σ_a.cx)*((σ_a.dx/2)+(σ_b.dx/2)), σ_b.cy))
      ua = σ_a.vector[0]; va = σ_a.vector[1];
      ub = σ_b.vector[0]; vb = σ_b.vector[1];


    d_a = x**2 + y**2; d_b = a**2 + (Y-y)**2
    θ_a = ShipSpeed**2 - ua**2 - va**2; θ_b = ShipSpeed**2 - ub**2 - vb**2   
    D_a = x*ua + y*va; D_b = a*ub + (Y-y)*vb
    X_a = np.sqrt(D_a**2 + θ_a*(d_a**2)); X_b = np.sqrt(D_b**2 + θ_b*(d_b**2))

    # Determine the travel-time between two grid-cell points
    TravelTime = ((X_a - D_a)/θ_a + (X_b - D_b)/θ_b)
    
    # Returning the point corrected back into Lat/Long

    return TravelTime


def distance(origin, destination):
    """
    Calculate the Haversine distance. Only bring into for smoothing.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

# Euclidian distance

class Mesh:
  def __init__(self,MeshInfo,verbrose=True):
    self.meshinfo = MeshInfo
    self.verbrose = verbrose
    
    self.xmin = self.meshinfo['Xmin']; self.xmax = self.meshinfo['Xmax']; self.dx   = self.meshinfo['dx']
    self.ymin = self.meshinfo['Ymin']; self.ymax = self.meshinfo['Ymax']; self.dy   = self.meshinfo['dy']



    # Generating the Initial Mesh
    if self.verbrose:
          print('\n==================================================')
          print('=============== Initialising Mesh ================')
          print('==================================================')
          print('Xmin={},Xmax={},dX={}  Ymin={},Ymax={},dY={}'.format(self.xmin,self.xmax,self.dx,self.ymin,self.ymax,self.dy))
    self.cells = []
    for xx in np.arange(self.xmin,self.xmax,self.dx):
      for yy in np.arange(self.ymin,self.ymax,self.dy):
          self.cells.append(CellBox(xx,yy,self.dx,self.dy))

    # Running Cell Splitting for Land
    self._isLand()


  def _isLand(self):
      X = self.meshinfo['CoastMask']['X']
      Y = self.meshinfo['CoastMask']['Y']
      V = self.meshinfo['CoastMask']['Mask']

      # Defining the initial boolean land mask
      if self.verbrose:
          print('\n==================================================')
          print('============ GRIDDING LAND INFORMATION ===========')
          print('==================================================')

          print('~~~~~~~~ Initialising Original Mask Array ~~~~~~~~')
      for cc in range(len(self.cells)):
         self.cells[cc].isLand = V.flatten()[np.argmin(np.sqrt((X.flatten()-self.cells[cc].cx)**2 + (Y.flatten()-self.cells[cc].cy)**2))]

      # -- Splitting the cells depending on Coastline
      if self.meshinfo['CoastMask']['Split']:
         if self.verbrose:
            print('~~~~~~~~ Splitting Mask array based on Coast ~~~~~~~~\n-----> Max Depth={}'.format(self.meshinfo['CoastMask']['SplitDepth']))
         self._cells = []
         while self.cells:
            Points,Pointsdx,Pointsdy = self.cells[0]._splitDomain()
            if (self.cells[0]._leafDepth + 1) > self.meshinfo['CoastMask']['SplitDepth']:
              self._cells.append(self.cells[0])
              self.cells.pop(0)
              continue              
            tmpCells = []; runningValues=0;
            for pt in Points:
              tmpCell = CellBox(pt[0],pt[1],Pointsdx,Pointsdy)
              tmpCell._leafDepth = self.cells[0]._leafDepth + 1
              # Determine if centre on any corners have land in
              Values = np.mean(V.flatten()[ (X.flatten()>=tmpCell.x) & (X.flatten()<=(tmpCell.x+tmpCell.dx)) & (Y.flatten()>=tmpCell.y) & (Y.flatten()<=(tmpCell.y+tmpCell.dy))])
              tmpCell.isLand = bool(int(Values))
              runningValues += Values
              tmpCells.append(tmpCell)
            runningValues = runningValues/4.

            if (runningValues == 0) or (runningValues == 1):
              self._cells.append(self.cells[0])
              self.cells.pop(0)
            else:
              self.cells.pop(0)
              self.cells = self.cells + tmpCells 
         self.cells = self._cells.copy(); self._cells=None;


  def IceInformation(self):
      X = self.meshinfo['IceExtent']['X']
      Y = self.meshinfo['IceExtent']['Y']
      V = self.meshinfo['IceExtent']['Values']

      # Defining the initial boolean land mask
      if self.verbrose:
          print('\n==================================================')
          print('============ GRIDDING ICE INFORMATION ===========')
          print('==================================================')

          print('~~~~~~~~ Initialising Original Mask Array ~~~~~~~~')
      for cc in range(len(self.cells)):
         self.cells[cc].value = V.flatten()[np.argmin(np.sqrt((X.flatten()-self.cells[cc].cx)**2 + (Y.flatten()-self.cells[cc].cy)**2))]

      # -- Splitting the cells depending on Coastline
      if self.meshinfo['IceExtent']['Split']:
         if self.verbrose:
            print('~~~~~~~~  Splitting Mask array based on Ice Content ~~~~~~~~\n-----> Max Depth={}'.format(self.meshinfo['IceExtent']['SplitDepth']))
         self._cells = []
         while self.cells:
            Points,Pointsdx,Pointsdy = self.cells[0]._splitDomain()
            if ((self.cells[0]._leafDepth + 1) > self.meshinfo['IceExtent']['SplitDepth'])  or self.cells[0].isLand:
              self._cells.append(self.cells[0])
              self.cells.pop(0)
              continue              
            tmpCells = []; runningValues= [];
            for pt in Points:
              tmpCell = CellBox(pt[0],pt[1],Pointsdx,Pointsdy)
              tmpCell._leafDepth = self.cells[0]._leafDepth + 1
              # Determine if centre on any corners have land in
              Value = np.mean(V.flatten()[ (X.flatten()>=tmpCell.x) & (X.flatten()<=(tmpCell.x+tmpCell.dx)) & (Y.flatten()>=tmpCell.y) & (Y.flatten()<=(tmpCell.y+tmpCell.dy))])
              tmpCell.value = Value
              runningValues.append(Value)
              tmpCells.append(tmpCell)
            runningValues = np.array(runningValues)

            if self.meshinfo['IceExtent']['SplitDiff'] >= (runningValues.max() - runningValues.min()):
              self._cells.append(self.cells[0])
              self.cells.pop(0)
            else:
              self.cells.pop(0)
              self.cells = self.cells + tmpCells 
         self.cells = self._cells.copy(); self._cells=None;

  def VectorInformation(self):
      X  = self.meshinfo['Currents']['X']
      Y  = self.meshinfo['Currents']['Y']
      Vx = self.meshinfo['Currents']['Vx']
      Vy = self.meshinfo['Currents']['Vy']
      if self.verbrose:
          print('\n==================================================')
          print('============ GRIDDING Current Data ===============')
          print('==================================================')
      for cc in range(len(self.cells)):
          if self.cells[cc].isLand:
            continue
          gVx = Vx.flatten()[ (X.flatten()>=self.cells[cc].x) & (X.flatten()<=(self.cells[cc].x+self.cells[cc].dx)) & (Y.flatten()>=self.cells[cc].y) & (Y.flatten()<=(self.cells[cc].y+self.cells[cc].dy))]
          gVy = Vy.flatten()[ (X.flatten()>=self.cells[cc].x) & (X.flatten()<=(self.cells[cc].x+self.cells[cc].dx)) & (Y.flatten()>=self.cells[cc].y) & (Y.flatten()<=(self.cells[cc].y+self.cells[cc].dy))]
          self.cells[cc].vector = np.array([np.nanmean(gVx),np.nanmean(gVy)])


  def NearestNeighbours(self,ii):
      # ======= Inspecting Close to Grid cells
      neighbours = []
      for jj in range(len(self.cells)): 
        if (ii == jj):
            continue
        # Looping over each independent Cell
        Xmin = self.cells[ii].x; Xmax = self.cells[ii].x+self.cells[ii].dx
        Ymin = self.cells[ii].y; Ymax = self.cells[ii].y+self.cells[ii].dy
        touchingBool = ((Xmin <= np.array(self.cells[jj]._bounding_points)[:,0]) & (np.array(self.cells[jj]._bounding_points)[:,0] <= Xmax) & (Ymin <= np.array(self.cells[jj]._bounding_points)[:,1]) & (np.array(self.cells[jj]._bounding_points)[:,1] <= Ymax)).any()
        if touchingBool == True:
            neighbours.append(jj)
      return neighbours

  def NewtonianDistance(self,ii):
      neighbours = self.NearestNeighbours(ii)
      points     = np.zeros((len(neighbours),2))
      TravelTime = np.zeros((len(neighbours)))

      for jj_id,jj in enumerate(neighbours):
        σ_a = self.cells[ii]
        σ_b = self.cells[jj]


        # ------ Move on if is Land or Thickness too large ------
        if (self.cells[jj].value >= self.meshinfo['IceExtent']['MaxProportion']) or self.cells[jj].isLand:
          xp=None;yp=None; TT=np.inf
          TravelTime[jj_id] = TT
          points[jj_id,:]   = np.array([xp,yp])
          continue

        # ------ Determining Distances if cell is plausible
        diff_x = σ_b.cx - σ_a.cx; diff_y = σ_b.cy - σ_a.cy

        # Longitude case
        if (abs(diff_x) > σ_a.dx/2) and (abs(diff_y) < σ_a.dy/2):
          try:
            θ  = np.arctan((σ_b.cy - σ_a.cy)/(σ_b.cx - σ_a.cx));
            y_init = np.tan(θ)*(σ_a.dx/2)
            y_opt = optimize.newton(f,y_init,args=(σ_a,σ_b,self.meshinfo['VehicleInfo']['Speed'],0))
            xp = σ_a.cx+np.sign(σ_b.cx-σ_a.cx)*σ_a.dx/2
            yp = σ_a.cy+y_opt
            TT = T(y_opt,σ_a,σ_b,self.meshinfo['VehicleInfo']['Speed'],0)
          except:
            xp=None;yp=None;TT=np.inf

        # Latitude case
        elif (abs(diff_x) < σ_a.dx/2) and (abs(diff_y) > σ_a.dy/2):
          try:
            θ  = np.arctan((σ_b.cx - σ_a.cx)/(σ_b.cy - σ_a.cy));
            y_init = np.tan(θ)*(σ_a.dy/2)
            y_opt = optimize.newton(f,y_init,args=(σ_a,σ_b,self.meshinfo['VehicleInfo']['Speed'],1))
            xp = σ_a.cx+y_opt
            yp = σ_a.cy+np.sign(σ_b.cy-σ_a.cy)*σ_a.dy/2
            TT = T(y_opt,σ_a,σ_b,self.meshinfo['VehicleInfo']['Speed'],0)
          except:
            xp=None;yp=None;TT=np.inf
        
        # Edge-case
        else:
          y_opt = np.sign((σ_b.cy - σ_a.cy))*(σ_a.dy/2)
          TT = T(y_opt,σ_a,σ_b,self.meshinfo['VehicleInfo']['Speed'],0)
          xp = np.sign((σ_b.cx - σ_a.cx))*(σ_a.dx/2) + σ_a.cx
          yp = np.sign((σ_b.cy - σ_a.cy))*(σ_a.dy/2) + σ_a.cy


        # -- Clipping the values to remain inbox
        if type(xp) != type(None):
          xp = np.clip(xp,σ_b.cx-σ_b.dx/2,σ_b.cx+σ_b.dx/2)
          yp = np.clip(yp,σ_b.cy-σ_b.dy/2,σ_b.cy+σ_b.dy/2)

        TravelTime[jj_id] = TT
        points[jj_id,:]   = np.array([xp,yp])

      return neighbours,points,TravelTime