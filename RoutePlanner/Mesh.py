import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy import optimize
import math
from matplotlib.patches import Polygon

from RoutePlanner.CellBox import CellBox

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