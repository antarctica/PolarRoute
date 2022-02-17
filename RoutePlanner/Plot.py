

from pkgutil import walk_packages
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import numpy as np

def Mesh(self,figInfo=None,currents=False,return_ax=False,iceThreshold=None):
    """
        plots this cellGrid for display.

        TODO - requires reworking as part of the plotting work-package
    """
    if type(figInfo) == type(None):
        fig,ax = plt.subplots(1,1,figsize=(15,10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('lightblue')
    else:
        fig,ax = figInfo

    for cellBox in self.cellBoxes:
        if cellBox.containsLand():
            ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = True, facecolor='mediumseagreen'))
            ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = False, edgecolor='gray'))
            continue


        iceArea = cellBox.iceArea()
        if iceArea >= 0.8:
            ax.add_patch(Polygon(cellBox.getBounds(),closed=True,fill=True,color='White'))
            ax.add_patch(Polygon(cellBox.getBounds(),closed=True,fill=True,color='Pink',alpha=0.4))
            ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = False,edgecolor='gray'))
        elif not np.isnan(iceArea):
            ax.add_patch(Polygon(cellBox.getBounds(),closed=True,fill=True,color='White',alpha=iceArea))
            ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = False,edgecolor='gray'))
        else:
            ax.add_patch(cellBox.getPolygon())

        if currents:
            ax.quiver((cellBox.long+cellBox.width/2),(cellBox.lat+cellBox.height/2),cellBox.getuC()*1000,cellBox.getvC()*1000,scale=2,width=0.002,color='gray')

    ax.set_xlim(self._longMin, self._longMax)
    ax.set_ylim(self._latMin, self._latMax)

    if return_ax:
        return ax


def MeshNeighbours(cellGrid,Xpoint,Ypoint,figInfo=None,return_ax=False):

    if type(figInfo) == type(None):
        fig,ax = plt.subplots(1,1,figsize=(15,10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('lightblue')
    else:
        fig,ax = figInfo

    ax = Mesh(cellGrid,figInfo=[fig,ax],return_ax=True)
    cell  = cellGrid.getCellBox(Xpoint,Ypoint)
    neigh = cellGrid.getNeightbours(cell)
    for ncell_indx,ncell in neigh.iterrows():
        ax.add_patch(Polygon(ncell['Cell'].getBounds(), closed = True, fill = False, color = 'Red'))
        ax.scatter(ncell['Cp'][0],ncell['Cp'][1],50,'b')
        ax.scatter(ncell['Cell'].cx,ncell['Cell'].cy,50,'r')
        ax.text(ncell['Cell'].cx+0.1,ncell['Cell'].cy+0.1,ncell['Case'])
    ax.add_patch(Polygon(cell.getBounds(), closed = True, fill = False, color = 'Black'))
    ax.scatter(cell.cx,cell.cy,50,'k')
    ax.scatter(Xpoint,Ypoint,50,'m')

    # Add in the xlims,ylims to neighbour grid cells

    if return_ax:
        return ax



def Paths(cellGrid,Paths,routepoints=False,figInfo=None,return_ax=False,Waypoints=None):
        if type(figInfo) == type(None):
            fig,ax = plt.subplots(1,1,figsize=(15,10))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('lightblue')
        else:
            fig,ax = figInfo

        ax = Mesh(cellGrid,figInfo=[fig,ax],return_ax=True)

        for Path in Paths:
            if Path['Time'] == np.inf:
                continue
            Points = np.array(Path['Path']['Points'])
            if routepoints:
                ax.plot(Points[:,0],Points[:,1],linewidth=1.0,color='k')
                ax.scatter(Points[:,0],Points[:,1],30,zorder=99,color='k')
            else:
                ax.plot(Points[:,0],Points[:,1],linewidth=1.0,color='k')


        if type(Waypoints) != type(None):
            ax.scatter(Waypoints['Long'],Waypoints['Lat'],50,marker='^',color='k')

        if return_ax:
            return ax
