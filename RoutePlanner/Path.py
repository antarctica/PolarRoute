import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy import optimize
import math
from matplotlib.patches import Polygon

from RoutePlanner.CellBox import CellBox
from RoutePlanner.Mesh import Mesh

def dijkstra(ms, idx):
  #Initialising a column array of all the indexs
  DijkstraInfo = {}
  DijkstraInfo['CellIndex']      = np.arange(len(ms.cells))
  DijkstraInfo['Cost']           = np.full((len(ms.cells)),np.inf)
  DijkstraInfo['PositionLocked'] = np.zeros((len(ms.cells)),dtype=bool)
  DijkstraInfo['Paths']          = {}
  for djk in range(len(ms.cells)):
    DijkstraInfo['Paths'][djk] = [idx]
  DijkstraInfo['Cost'][idx]           = 0.0
  while (DijkstraInfo['PositionLocked'] == False).any():
    # Determining the argument with the next lowest value and hasn't been visited
    idx = DijkstraInfo['CellIndex'][(DijkstraInfo['PositionLocked']==False)][np.nanargmin(DijkstraInfo['Cost'][(DijkstraInfo['PositionLocked']==False)])]
    # Finding the cost of the nearest neighbours
    Neighbour_index,Points,TT = ms.NewtonianDistance(idx)
    Neighbour_cost = TT + DijkstraInfo['Cost'][idx]
    # Determining if the visited time is visited    
    for jj_v,jj in enumerate(Neighbour_index):
      if Neighbour_cost[jj_v] < DijkstraInfo['Cost'][jj]:
        DijkstraInfo['Cost'][jj]  = Neighbour_cost[jj_v]
        DijkstraInfo['Paths'][jj] = DijkstraInfo['Paths'][idx] + [jj]
    # Defining the graph point as visited
    DijkstraInfo['PositionLocked'][idx] = True
 
  return DijkstraInfo