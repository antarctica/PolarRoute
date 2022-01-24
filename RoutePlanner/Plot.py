import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy import optimize
import math
from matplotlib.patches import Polygon

from RoutePlanner.CellBox import CellBox
from RoutePlanner.Mesh import Mesh


def PlotMesh(ms):
  from matplotlib.patches import Polygon
  fig,ax = plt.subplots(1,1,figsize=(15,10))
  fig.patch.set_facecolor('white')

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

  ax.set_xlim([ms.meshinfo['Xmin'],ms.meshinfo['Xmax']])
  ax.set_ylim([ms.meshinfo['Ymin'],ms.meshinfo['Ymax']])

def OptimisedPaths(ms,optimizer):
    from matplotlib.patches import Polygon
    fig,ax = plt.subplots(1,1,figsize=(15,10))
    fig.patch.set_facecolor('white')

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
            ax.add_patch(Polygon(Bounds, closed=True, fill= True, color='Blue',alpha=0.4))

        else:
            ax.add_patch(Polygon(Bounds, closed=True,fill=False,edgecolor='Gray'))
            #ax.quiver(cell.cx,cell.cy,cell.vector[0],cell.vector[1])

    ax.set_xlim([ms.meshinfo['Xmin'],ms.meshinfo['Xmax']])
    ax.set_ylim([ms.meshinfo['Ymin'],ms.meshinfo['Ymax']])


    # Constructing the cell paths information
    for indx in range(len(optimizer.Paths['Path'])):
        Points = np.array(optimizer.Paths['Path'][indx])
        ax.plot(Points[:,0],Points[:,1],linewidth=1.0,color='k')
        ax.scatter(Points[:,0],Points[:,1],15,marker='o',color='k')

    # Plotting Waypoints
    ax.scatter(optimizer.OptInfo['WayPoints']['Long'],optimizer.OptInfo['WayPoints']['Lat'],100,marker='^',color='b')
    for idx,wpt in enumerate(optimizer.OptInfo['WayPoints'].iterrows()):
        Long = wpt[1]['Long']
        Lat  = wpt[1]['Lat']
        Name = wpt[1]['Name']
        ax.text(Long,Lat,Name)
