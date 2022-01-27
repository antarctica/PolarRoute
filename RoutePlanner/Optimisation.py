import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy import optimize
import math
from matplotlib.patches import Polygon
import copy
import random

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from RoutePlanner.CellBox_JS import CellBox
from RoutePlanner.Mesh import Mesh
from RoutePlanner.Function import NewtonianDistance, SmoothedNewtonianDistance

class TravelTime:
    def __init__(self,Mesh,OptInfo,CostFunc=NewtonianDistance):
        # Load in the current cell structure & Optimisation Info
        self.Mesh    = copy.copy(Mesh)
        self.OptInfo = copy.copy(OptInfo)


        self.CostFunc = CostFunc

        # ====== Waypoints ======        
        # Dropping waypoints outside domain
        self.OptInfo['WayPoints'] = self.OptInfo['WayPoints'][(self.OptInfo['WayPoints']['Long'] >= self.Mesh.xmin) & (self.OptInfo['WayPoints']['Long'] <= self.Mesh.xmax) & (self.OptInfo['WayPoints']['Lat'] <= self.Mesh.ymax) & (self.OptInfo['WayPoints']['Lat'] >= self.Mesh.ymin)] 

        # Initialising Waypoints positions and cell index
        self.OptInfo['WayPoints']['Index'] = np.nan
        for idx,wpt in enumerate(self.OptInfo['WayPoints'].iterrows()):
            Long = wpt[1]['Long']
            Lat  = wpt[1]['Lat']
            for index, cell in enumerate(self.Mesh.cells):
                if (Long>=cell.x) and (Long<=(cell.x+cell.dx)) and (Lat>=cell.y) and (Lat<=(cell.y+cell.dy)):
                    break
            self.OptInfo['WayPoints']['Index'].iloc[idx] = index
            self.Mesh.cells[index]._define_waypoints((Long,Lat))
        

        # ====== Dijkstra Formulation ======
        # Initialising the Dijkstra Info Dictionary
        self.DijkstraInfo = {}
        for wpt in self.OptInfo['WayPoints'].iterrows():
            wpt_name  = wpt[1]['Name']
            wpt_index = int(wpt[1]['Index'])
            self.DijkstraInfo[wpt_name] = {}
            self.DijkstraInfo[wpt_name]['CellIndex']       = np.arange(len(self.Mesh.cells))
            self.DijkstraInfo[wpt_name]['TotalCost']       = np.full((len(self.Mesh.cells)),np.inf)
            self.DijkstraInfo[wpt_name]['PositionLocked']  = np.zeros((len(self.Mesh.cells)),dtype=bool)
            self.DijkstraInfo[wpt_name]['Path']            = {}
            self.DijkstraInfo[wpt_name]['Path']['FullPath']         = {}
            self.DijkstraInfo[wpt_name]['Path']['CrossingPoints']   = {}
            self.DijkstraInfo[wpt_name]['Path']['CentroidPoints']   = {}
            self.DijkstraInfo[wpt_name]['Path']['CellIndex']        = {}
            self.DijkstraInfo[wpt_name]['Path']['Cost']             = {}
            for djk in range(len(self.Mesh.cells)):
                self.DijkstraInfo[wpt_name]['Path']['FullPath'][djk] = []
                self.DijkstraInfo[wpt_name]['Path']['CrossingPoints'][djk] = []
                self.DijkstraInfo[wpt_name]['Path']['CentroidPoints'][djk] = []
                self.DijkstraInfo[wpt_name]['Path']['CellIndex'][djk]      = [wpt_index]
                self.DijkstraInfo[wpt_name]['Path']['Cost'][djk]           = [np.inf]
            self.DijkstraInfo[wpt_name]['TotalCost'][wpt_index]       = 0.0
            self.DijkstraInfo[wpt_name]['Path']['Cost'][wpt_index] = [0.0]

        # ====== Path Information ======
        # Initialising Path & Path Smoothed defintion
        self.Paths         = {}
        self.SmoothedPaths = {}
        self._paths_smoothed = False

    def NeighbourCost(self,index):
        '''
        Function for computing the shortest travel-time from a cell to its neighbours by applying the Newtonian method for optimisation
        
        Inputs:
        index - Index of the cell to process
        
        Output:

        Bugs/Alterations:
            - If corner of cell is land in adjacent cell then also return 'inf'
        '''
        # Determining the nearest neighbour index for the cell
        neighbours = self.Mesh.NearestNeighbours(index)

        # Defining the vehicle speed
        s    = self.OptInfo['VehicleInfo']['Speed']

        # Creating Blank travel-time and crossing point array
        TravelTime  = np.zeros((len(neighbours)))
        CrossPoints = np.zeros((len(neighbours),2))
        CellPoints  = np.zeros((len(neighbours),2))

        # Defining the starting cell
        Cell_s = self.Mesh.cells[index]

        # Looping over all the nearest neighbours 
        for lp_index, neighbour_index in enumerate(neighbours):
            # Determining the cell for the neighbour
            Cell_n = self.Mesh.cells[neighbour_index]

            # Set travel-time to infinite if neighbour is land or ice-thickness is too large.
            if (Cell_n.value >= self.Mesh.meshinfo['IceExtent']['MaxProportion']) or (Cell_s.value >= self.Mesh.meshinfo['IceExtent']['MaxProportion']) or (Cell_n.isLand) or (Cell_s.isLand):
                TravelTime[lp_index] = np.inf
                continue

            TravelTime[lp_index], CrossPoints[lp_index,:], CellPoints[lp_index,:] = self.CostFunc(Cell_s,Cell_n,s).value()

        return neighbours, TravelTime, CrossPoints, CellPoints


    def Dijkstra2Path(self):
        self.Paths = []
        for wpt_a in self.OptInfo['WayPoints'].iterrows():
            wpt_a_name  = wpt_a[1]['Name']; wpt_a_index = int(wpt_a[1]['Index']); wpt_a_loc   = [[wpt_a[1]['Long'],wpt_a[1]['Lat']]]
            for wpt_b in self.OptInfo['WayPoints'].iterrows():
                wpt_b_name  = wpt_b[1]['Name']; wpt_b_index = int(wpt_b[1]['Index']); wpt_b_loc   = [[wpt_b[1]['Long'],wpt_b[1]['Lat']]]
                if not wpt_a_name == wpt_b_name:
                    Path = {}
                    Path['from']                   = wpt_a_name
                    Path['to']                     = wpt_b_name
                    Path['TotalCost']              = self.DijkstraInfo[wpt_a_name]['TotalCost'][wpt_b_index]  
                    Path['Path']                   = {}
                    Path['Path']['FullPath']       = np.array(wpt_a_loc+self.DijkstraInfo[wpt_a_name]['Path']['FullPath'][wpt_b_index]+wpt_b_loc)
                    Path['Path']['CrossingPoints'] = np.array(self.DijkstraInfo[wpt_a_name]['Path']['CrossingPoints'][wpt_b_index])
                    Path['Path']['CentroidPoints'] = np.array(wpt_a_loc+self.DijkstraInfo[wpt_a_name]['Path']['CentroidPoints'][wpt_b_index]+wpt_b_loc)
                    Path['Path']['CellIndex']      = np.array(self.DijkstraInfo[wpt_a_name]['Path']['CellIndex'][wpt_b_index])
                    Path['Path']['Cost']           = np.array(self.DijkstraInfo[wpt_a_name]['Path']['Cost'][wpt_b_index])
                    self.Paths.append(Path)

    def Dijkstra(self,verbrose=False):
        '''
        Determining the shortest path between all waypoints
        '''
        for wpt in self.OptInfo['WayPoints'].iterrows():
            wpt_name  = wpt[1]['Name']
            if verbrose:
                print('=== Processing Waypoint = {} ==='.format(wpt_name))

            # Loop over all the points until all waypoints are visited
            while (self.DijkstraInfo[wpt_name]['PositionLocked'][np.array(self.OptInfo['WayPoints']['Index'].astype(int))] == False).any():
                # Determining the argument with the next lowest value and hasn't been visited
                idx = self.DijkstraInfo[wpt_name]['CellIndex'][(self.DijkstraInfo[wpt_name]['PositionLocked']==False)][np.argmin(self.DijkstraInfo[wpt_name]['TotalCost'][(self.DijkstraInfo[wpt_name]['PositionLocked']==False)])]
                # Finding the cost of the nearest neighbours
                Neighbour_index,TT,CrossPoints,CellPoints = self.NeighbourCost(idx)
                Neighbour_cost     = TT + self.DijkstraInfo[wpt_name]['TotalCost'][idx]
                # Determining if the visited time is visited    
                for jj_v,jj in enumerate(Neighbour_index):
                    if Neighbour_cost[jj_v] < self.DijkstraInfo[wpt_name]['TotalCost'][jj]:
                        self.DijkstraInfo[wpt_name]['TotalCost'][jj]              = Neighbour_cost[jj_v]
                        self.DijkstraInfo[wpt_name]['Path']['FullPath'][jj]       = self.DijkstraInfo[wpt_name]['Path']['FullPath'][idx]  + [[CrossPoints[jj_v,0],CrossPoints[jj_v,1]]] + [[CellPoints[jj_v,0],CellPoints[jj_v,1]]]
                        self.DijkstraInfo[wpt_name]['Path']['CrossingPoints'][jj] = self.DijkstraInfo[wpt_name]['Path']['CrossingPoints'][idx] + [[CrossPoints[jj_v,0],CrossPoints[jj_v,1]]]
                        self.DijkstraInfo[wpt_name]['Path']['CentroidPoints'][jj] = self.DijkstraInfo[wpt_name]['Path']['CentroidPoints'][idx] + [[CellPoints[jj_v,0],CellPoints[jj_v,1]]]
                        self.DijkstraInfo[wpt_name]['Path']['CellIndex'][jj]      = self.DijkstraInfo[wpt_name]['Path']['CellIndex'][idx] + [jj]
                        self.DijkstraInfo[wpt_name]['Path']['Cost'][jj]           = self.DijkstraInfo[wpt_name]['Path']['Cost'][idx] + [Neighbour_cost[jj_v]]
                # Defining the graph point as visited
                self.DijkstraInfo[wpt_name]['PositionLocked'][idx] = True

        # Chaning Dijkstra Information to Paths
        self.Dijkstra2Path()

    def PathSmoothing(self):
        '''
            Given a series of pathways smooth without centroid locations using great circle smoothing
        '''

        if self._paths_smoothed:
            print('Path smmothing has already been run, do you want to smooth again ?\nIf so re-run this function')
        else:
            self._paths_smoothed = True

        self.SmoothedPaths = self.Paths.copy()

        # Looping over all the optimised paths
        for Path in self.SmoothedPaths:
            path_fullpath   = Path['Path']['FullPath']
            path_edges      = Path['Path']['CrossingPoints']
            path_centroid   = Path['Path']['CentroidPoints']
            path_cellIndex  = Path['Path']['CellIndex']
            path_cost       = Path['Path']['Cost']

            for idxpt in range(len(path_edges)-2):
                pt_sp  = path_edges[idxpt]
                pt_cp  = path_edges[idxpt+1]
                pt_np  = path_edges[idxpt+2]
                ind_sp = path_cellIndex[idxpt]
                ind_np = path_cellIndex[idxpt+1]

                # Determine Starting & Ending cells
                Cell_s    = copy.copy(self.Mesh.cells[ind_sp])
                Cell_s.cx = pt_sp[0]; Cell_s.cy = pt_sp[1]
                Cell_s.dxp = ((Cell_s.x + Cell_s.dx) - Cell_s.cx); Cell_s.dxm = (Cell_s.cx - Cell_s.x)
                Cell_s.dyp = ((Cell_s.y + Cell_s.dy) - Cell_s.cy); Cell_s.dym = (Cell_s.cy - Cell_s.y)
                Cell_n    = copy.copy(self.Mesh.cells[ind_np])
                Cell_n.cx = pt_np[0]; Cell_n.cy = pt_np[1]
                Cell_n.dxp = ((Cell_n.x + Cell_n.dx) - Cell_n.cx); Cell_n.dxm = (Cell_n.cx - Cell_n.x)
                Cell_n.dyp = ((Cell_n.y + Cell_n.dy) - Cell_n.cy); Cell_n.dym = (Cell_n.cy - Cell_n.y)
                TravelTime, CrossingPoint, delA = self.CostFunc(Cell_s,Cell_n,self.OptInfo['VehicleInfo']['Speed']).value()
                
                # Clipping so on reciever grid
                # CrossingPoint[0] = np.clip(CrossingPoint[0],Cell_n.x,(Cell_n.x+Cell_n.dx))
                # CrossingPoint[1] = np.clip(CrossingPoint[1],Cell_n.y,(Cell_n.y+Cell_n.dy))


                path_edges[idxpt+1] = CrossingPoint
                path_cost[idxpt+1]  = path_cost[idxpt] + TravelTime

            # Now determining if you can get rid of a point 
            Path['Path']['CrossingPoints']   = path_edges
            Path['Path']['FullPath']         = np.concatenate([path_fullpath[0,:][None,:],np.array(path_edges),path_fullpath[-1,:][None,:]])