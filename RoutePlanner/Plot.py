import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy import optimize
import math
from matplotlib.patches import Polygon

from RoutePlanner.CellBox import CellBox
from RoutePlanner.Mesh import Mesh


def PlotMesh(ms,figInfo=None):
    from matplotlib.patches import Polygon

    if type(figInfo) == type(None):
      fig,ax = plt.subplots(1,1,figsize=(15,10))
      fig.patch.set_facecolor('white')
    else:
      fig,ax = figInfo

    Vals = []
    for cell in ms.cells:
        Bounds = [[cell.x,cell.y],
                            [cell.x,cell.y+cell.dy],
                            [cell.x+cell.dx,cell.y+cell.dy],
                            [cell.x+cell.dx,cell.y],
                            [cell.x,cell.y]]

        ax.quiver((cell.x+cell.dx/2),(cell.y+cell.dy/2),cell.vector[0],cell.vector[1],scale=2,width=0.002,color='gray')

        if cell.isLand:
            ax.add_patch(Polygon(Bounds, closed=True,fill=True,color='Green',edgecolor='Gray'))
        else:
            ax.add_patch(Polygon(Bounds, closed=True,fill=True,color='Blue',alpha=cell.value))
            if cell.value > ms.meshinfo['IceExtent']['MaxProportion']:
                qp = ax.add_patch(Polygon(Bounds, closed=True, fill= False))
                qp.set_hatch('/')
        
        ax.add_patch(Polygon(Bounds, closed=True,fill=False,color='gray'))

    ax.set_xlim([ms.meshinfo['Xmin'],ms.meshinfo['Xmax']])
    ax.set_ylim([ms.meshinfo['Ymin'],ms.meshinfo['Ymax']])

def OptimisedPaths(ms,optimizer,Paths,figInfo=None,routepoints=True):
    from matplotlib.patches import Polygon
    if type(figInfo) == type(None):
      fig,ax = plt.subplots(1,1,figsize=(15,10))
      fig.patch.set_facecolor('white')
    else:
      fig,ax = figInfo


    PlotMesh(ms,figInfo=[fig,ax])

    # Constructing the cell paths information
    for Path in Paths:
        if Path['TotalCost'] == np.inf:
          continue
        Points = np.array(Path['Path']['FullPath'])
        ax.plot(Points[:,0],Points[:,1],linewidth=1.0,color='k')
        if routepoints:
          ax.scatter(Points[:,0],Points[:,1],15,marker='o',color='k')


    # Plotting Waypoints
    ax.scatter(optimizer.OptInfo['WayPoints']['Long'],optimizer.OptInfo['WayPoints']['Lat'],100,marker='^',color='r',zorder=100)
    for wpt in optimizer.OptInfo['WayPoints'].iterrows():
        Long = wpt[1]['Long']
        Lat  = wpt[1]['Lat']
        Name = wpt[1]['Name']
        ax.text(Long,Lat,Name,color='r',zorder=100)

def RandomPaths(ms,optimizer,WaypointName,indices=100,figInfo=None):
    from matplotlib.patches import Polygon
    if type(figInfo) == type(None):
      fig,ax = plt.subplots(1,1,figsize=(15,10))
      fig.patch.set_facecolor('white')
    else:
      fig,ax = figInfo

    Paths = optimizer.DijkstraInfo[WaypointName]

    PlotMesh(ms,figInfo=[fig,ax])

    # If no indices given then return random array
    if type(indices) == int:
        idx = np.random.randint(0,high=len(ms.cells),size=indices)
    else:
        idx = indices
  
    # Constructing the cell paths information
    for indx in idx:
        if Paths['Cost'][indx] == np.inf:
          continue
        Points = np.concatenate([np.array(optimizer.OptInfo['WayPoints'][optimizer.OptInfo['WayPoints']['Name'] == WaypointName][['Long','Lat']]),
                np.array(Paths['Paths'][indx]),
                np.array([ms.cells[indx].cx,ms.cells[indx].cy])[None,:]])

        ax.plot(Points[:,0],Points[:,1],linewidth=1.0,color='k')
        ax.scatter(Points[-1,0],Points[-1,1],50,marker='^',color='k')

    # Plotting Waypoints
    ax.scatter(optimizer.OptInfo['WayPoints']['Long'],optimizer.OptInfo['WayPoints']['Lat'],100,marker='^',color='k')
    for wpt in optimizer.OptInfo['WayPoints'].iterrows():
        Long = wpt[1]['Long']
        Lat  = wpt[1]['Lat']
        Name = wpt[1]['Name']
        ax.text(Long,Lat,Name)

    # Plotting Key Waypoint Blue
    ax.scatter(optimizer.OptInfo['WayPoints'][optimizer.OptInfo['WayPoints']['Name'] == WaypointName]['Long'],
               optimizer.OptInfo['WayPoints'][optimizer.OptInfo['WayPoints']['Name'] == WaypointName]['Lat'],
               100,marker='^',color='r')
