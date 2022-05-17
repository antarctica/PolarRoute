'''
    FILL
'''
import numpy as np
import copy
import pandas as pd

import time
import multiprocessing as mp

from RoutePlanner.crossing import NewtonianDistance, NewtonianCurve
from RoutePlanner.CellBox import CellBox

from shapely.geometry import Polygon, Point
from shapely import wkt
import geopandas as gpd
import ast
import json


class SpeedFunctions:
    def __init__(self,config):
        '''
            FILL
        '''

        self.config =config


        self.neighbour_graph = pd.read_csv(self.config['Speed_Function']['Mesh_Input_Filename'])
        self.neighbour_graph['geometry'] = self.neighbour_graph['geometry'].apply(wkt.loads)
        self.neighbour_graph = gpd.GeoDataFrame(self.neighbour_graph,crs={'init': 'epsg:4326'}, geometry='geometry')


        # Removing land or thick-ice cells
        #self.neighbour_graph = self.neighbour_graph[(self.neighbour_graph['Land']==False) & (self.neighbour_graph['Ice Area'] <= self.config['Vehicle_Info']['MaxIceExtent'])]

        # Reformating the columns into corret type
        self.neighbour_graph['case'] = self.neighbour_graph['case'].apply(lambda x: ast.literal_eval(x))
        self.neighbour_graph['cell_info'] = self.neighbour_graph['cell_info'].apply(lambda x: ast.literal_eval(x))
        self.neighbour_graph['neighbourIndex'] = self.neighbour_graph['neighbourIndex'].apply(lambda x: ast.literal_eval(x))
        self.neighbour_graph['Vector'] = self.neighbour_graph['Vector'].apply(lambda x: ast.literal_eval(x))

        # Checking if Speed defined in file
        if 'Speed' not in self.neighbour_graph:
            self.neighbour_graph['Speed'] = self.config["Vehicle_Info"]["Speed"]


        self.speed()

        self.neighbour_graph.to_csv(self.config['Speed_Function']['Mesh_Output_Filename'],index=False)


    def ice_resistance(self):
        """
                Function to find the ice resistance force at a given speed in a given cell.

                Inputs:
                cell - Cell box object

                Outputs:
                resistance - Resistance force
        """
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

        hull = self.config['Vehicle_Info']['HullType']
        beam = self.config['Vehicle_Info']['Beam']
        kparam, bparam, nparam = hull_params[hull]
        gravity = 9.81  # m/s-2
        speed = self.config['Vehicle_Info']['Speed']*(5./18.)  # assume km/h and convert to m/s
        force_limit = speed/np.sqrt(gravity*self.neighbour_graph['Ice Area']/100*self.neighbour_graph['Ice Thickness'])
        resistance = 0.5*kparam*(force_limit**bparam)*self.neighbour_graph['Ice Density']*beam*self.neighbour_graph['Ice Thickness']*(speed**2)*((self.neighbour_graph['Ice Area']/100 )**nparam)
        return resistance

    def inverse_resistance(self):
        """
        Function to find the speed that keeps the ice resistance force below a given threshold.

        Inputs:
        force_limit - Force limit
        cell        - Cell box object

        Outputs:
        speed - Vehicle Speed
        """
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}
        force_limit = self.config['Vehicle_Info']['ForceLimit']

        hull = self.config['Vehicle_Info']['HullType']
        beam = self.config['Vehicle_Info']['Beam']
        kparam, bparam, nparam = hull_params[hull]
        gravity = 9.81  # m/s-2

        vexp = 2*force_limit/(kparam*self.neighbour_graph['Ice Density']*beam*self.neighbour_graph['Ice Thickness']*((self.neighbour_graph['Ice Area']/100)**nparam)*(gravity*self.neighbour_graph['Ice Thickness']*self.neighbour_graph['Ice Area']/100)**-(bparam/2))

        vms = vexp**(1/(2.0 + bparam))
        speed = vms*(18./5.)  # convert from m/s to km/h

        return speed

    def speed(self):
        '''
            FILL
        '''


        self.neighbour_graph['Speed'] = self.inverse_resistance()
        self.neighbour_graph['Speed'][self.neighbour_graph['Ice Area'] == 0.0] = self.config["Vehicle_Info"]["Speed"]
        self.neighbour_graph['Speed'][self.ice_resistance() < self.config['Vehicle_Info']['ForceLimit']] = self.config["Vehicle_Info"]["Speed"]

    def speed_simple(self):
        self.neighbour_graph['Speed'] = (1-np.sqrt(self.neighbour_graph['Ice Area']/100))*self.config['Vehicle_Info']['Speed']