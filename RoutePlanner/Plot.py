import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy import optimize
import math
from matplotlib.patches import Polygon

from RoutePlanner.CellBox import CellBox
from RoutePlanner.Mesh import Mesh

def PlotNeighbours(ms,CellIndx):
    fig,ax = plt.subplots(1,1,figsize=(15,10))
    for idx, cell in enumerate(np.take(ms.cells,np.array(ms.cells[CellIndx].neighbour['Index']))):
      Bounds = [[cell.x,cell.y],
                        [cell.x,cell.y+cell.dy],
                        [cell.x+cell.dx,cell.y+cell.dy],
                        [cell.x+cell.dx,cell.y],
                        [cell.x,cell.y]]
      if cell.isLand:
        ax.add_patch(Polygon(Bounds, closed=True,
                          fill=True,color='Green'))
      elif cell.value > ms.meshinfo['IceExtent']['MaxProportion']:
        ax.quiver(cell.cx,cell.cy,cell.vector[0],cell.vector[1])
        ax.add_patch(Polygon(Bounds, closed=True, fill= True, color='Blue',alpha=0.4))

      else:
        ax.add_patch(Polygon(Bounds, closed=True,fill=False))
        ax.quiver(cell.cx,cell.cy,cell.vector[0],cell.vector[1])


      ax.plot([ms.cells[CellIndx].cx,np.array(ms.cells[CellIndx].neighbour['CrossingPoints'])[idx,0],cell.cx],
              [ms.cells[CellIndx].cy,np.array(ms.cells[CellIndx].neighbour['CrossingPoints'])[idx,1],cell.cy],'k--')

      
      ax.annotate(ms.cells[CellIndx].neighbour['TravelTime'][idx],(cell.cx,cell.cy))
      ms.cells[0].neighbour

    for cell in [ms.cells[CellIndx]]:
      Bounds = [[cell.x,cell.y],
                        [cell.x,cell.y+cell.dy],
                        [cell.x+cell.dx,cell.y+cell.dy],
                        [cell.x+cell.dx,cell.y],
                        [cell.x,cell.y]]
      if cell.isLand:
        ax.add_patch(Polygon(Bounds, closed=True,
                          fill=True,color='Green'))
      elif cell.value > ms.meshinfo['IceExtent']['MaxProportion']:
        ax.quiver(cell.cx,cell.cy,cell.vector[0],cell.vector[1])
        ax.add_patch(Polygon(Bounds, closed=True, fill= True, color='Blue',alpha=0.4))

      else:
        ax.add_patch(Polygon(Bounds, closed=True,fill=False))
        ax.quiver(cell.cx,cell.cy,cell.vector[0],cell.vector[1])      

    



def PlotRegion(ms):
  from matplotlib.patches import Polygon
  fig,ax = plt.subplots(1,1,figsize=(15,10))

  vls = ms.meshinfo['IceExtent']['Values']
  vls[ms.meshinfo['IceExtent']['Mask']] = np.nan
  ax.pcolormesh(ms.meshinfo['IceExtent']['X'],ms.meshinfo['IceExtent']['Y'],vls,cmap='Reds',vmin=0,vmax=1.0)


  Vals = []
  for cell in ms.cells:
    Bounds = [[cell.x,cell.y],
                       [cell.x,cell.y+cell.dy],
                       [cell.x+cell.dx,cell.y+cell.dy],
                       [cell.x+cell.dx,cell.y],
                       [cell.x,cell.y]]
    if cell.isLand:
      ax.add_patch(Polygon(Bounds, closed=True,
                        fill=True,color='Green'))
    elif cell.value > ms.meshinfo['IceExtent']['MaxProportion']:
      #ax.quiver(cell.cx,cell.cy,cell.vector[0],cell.vector[1])
      ax.add_patch(Polygon(Bounds, closed=True, fill= True, color='Blue',alpha=0.4))

    else:
      ax.add_patch(Polygon(Bounds, closed=True,fill=False))
      #ax.quiver(cell.cx,cell.cy,cell.vector[0],cell.vector[1])

  ax.set_xlim([ms.meshinfo['Xmin'],ms.meshinfo['Xmax']])
  ax.set_ylim([ms.meshinfo['Ymin'],ms.meshinfo['Ymax']])




  ax.scatter(ms.meshinfo['WayPoints']['Long'],ms.meshinfo['WayPoints']['Lat'],100,marker='^',color='b',zorder=3)
  # for ii,txt in ms.meshinfo['WayPoints'].iterrows():
  #     ax.annotate(txt['Name'][:], (txt['Long'], txt['Lat']),color='b',zorder=3)


def PlotRegionPath(ms,DijkstraInfo,WayIndex):
  from matplotlib.patches import Polygon
  fig,ax = plt.subplots(1,1,figsize=(15,10))

  vls = ms.meshinfo['IceExtent']['Values']
  vls[ms.meshinfo['IceExtent']['Mask']] = np.nan
  ax.pcolormesh(ms.meshinfo['IceExtent']['X'],ms.meshinfo['IceExtent']['Y'],vls,cmap='Reds',vmin=0,vmax=1.0)


  Vals = []
  for cell in ms.cells:
    Bounds = [[cell.x,cell.y],
                       [cell.x,cell.y+cell.dy],
                       [cell.x+cell.dx,cell.y+cell.dy],
                       [cell.x+cell.dx,cell.y],
                       [cell.x,cell.y]]
    if cell.isLand:
      ax.add_patch(Polygon(Bounds, closed=True,
                        fill=True,color='Green',edgecolor='Gray'))
    elif cell.value > ms.meshinfo['IceExtent']['MaxProportion']:
      #ax.quiver(cell.cx,cell.cy,cell.vector[0],cell.vector[1])
      ax.add_patch(Polygon(Bounds, closed=True, fill= True, color='Blue',alpha=0.4,edgecolor='Gray'))

    else:
      ax.add_patch(Polygon(Bounds, closed=True,fill=False,edgecolor='Gray'))
      #ax.quiver(cell.cx,cell.cy,cell.vector[0],cell.vector[1])

  ax.set_xlim([ms.meshinfo['Xmin'],ms.meshinfo['Xmax']])
  ax.set_ylim([ms.meshinfo['Ymin'],ms.meshinfo['Ymax']])


  # Constructing the cell paths information



  for indx in WayIndex:
    cellind = DijkstraInfo['Paths'][indx]
    Points  = np.zeros((len(cellind),2))
    for jj in range(len(cellind)):
      Points[jj,:] = np.array([ms.cells[cellind[jj]].cx,ms.cells[cellind[jj]].cy])
    ax.plot(Points[:,0],Points[:,1],'--k',linewidth=1.0)
    ax.scatter(Points[0,0],Points[0,1],100,marker='s',color='blue')
    ax.scatter(Points[-1,0],Points[-1,1],100,marker='s',color='black')

  ax.scatter(ms.meshinfo['WayPoints']['Long'],ms.meshinfo['WayPoints']['Lat'],100,marker='^',color='b',zorder=3)
  # for ii,txt in ms.meshinfo['WayPoints'].iterrows():
  #     ax.annotate(txt['Name'][:], (txt['Long'], txt['Lat']),color='b',zorder=3)