import logging
import math
import numpy as np
import matplotlib.pyplot as plt

import json

from polar_route.mesh_generation.cellbox import CellBox


class Mesh:
    """
        Attributes:
            cellboxes (list<(CellBox)>): A list of CellBox objects forming the Mesh

            neighbour_graph (NeighbourGraph): A graphical representation of the adjacency
                relationship between CellBoxes in the Mesh. The neighbour_graph object conatin a dict
                of the form

               {\n
                    <CellBox id_1>: {\n
                        "1": [id_1,...,id_n],\n
                        "2": [id_1,...,id_n],\n
                        "3": [id_1,...,id_n],\n
                        "4": [id_1,...,id_n],\n
                        "-1": [id_1,...,id_n],\n
                        "-2": [id_1,...,id_n],\n
                        "-3": [id_1,...,id_n],\n
                        "-4": [id_1,...,id_n],\n
                    },\n
                    ...,\n
                    {\n
                        <CellBox id_n>: {\n
                            ...\n
                        }\n
                    }\n
               }

    """

    def __init__(self, boundary , cellboxes , neighbour_graph , max_split_depth):
        """
            Constructs a Mesh object from a given parameters.

            Args:
                boundary (Boundary): the geo-spatial/temporal boundaries (longtitude, latitude and time) of the Mesh \n
                cellboxes (list <CellBox>): a list of the cellboxes contained in the Mesh \n
                neghbour_graph (NeighbourGraph): a graphical representation of the adjacency \n
                relationship between CellBoxes in the Mesh \n      
        """
        self.boundary = boundary
        self.cellboxes = cellboxes
        self.neighbour_graph = neighbour_graph
        self.max_split_depth = max_split_depth
        self.config = {}

    # Functions for adding data to the Mesh

    def add_data_points(self, data_points):
        """
            takes a dataframe containing geospatial-temporal located values and assigns them to
            cellboxes within this Mesh.

            Args:
                data_points (DataFrame): a dataframe of datapoints to be added to the Mesh.
                    data_points is of the form \n
                    lat | long | (time)* | value_1 | ... | value_n
        """
        for cell_box in self.cellboxes:
            if isinstance(cell_box, CellBox):
                long_loc = data_points.loc[(data_points['long'] > cell_box.long) &
                                           (data_points['long'] <= (cell_box.long + cell_box.width))]
                lat_long_loc = long_loc.loc[(long_loc['lat'] > cell_box.lat) &
                                            (long_loc['lat'] <= (cell_box.lat + cell_box.height))]

                cell_box.add_data_points(lat_long_loc)

    # Functions for outputting the Mesh

    def set_config(self, config):
        self.config = config

    def get_config (self):
        return self.config

    def get_cellboxes (self):
        return self.cellboxes

    def set_cellboxes (self, cellboxes):
        self.cellboxes = cellboxes

    def get_cellboxes(self):
        """
            returns a list of dictionaries containing information about each cellbox
            in this Mesh.
            all cellboxes will include id, geometry, cx, cy, dcx, dcy

            Returns:
                cellboxes (list<dict>): a list of CellBoxes which form the Mesh.
                    CellBoxes are of the form -

                    {
                        "id": (string) ... \n
                        "geometry": (string) POLYGON(...), \n
                        "cx": (float) ..., \n
                        "cy": (float) ..., \n
                        "dcx": (float) ..., \n
                        "dcy": (float) ..., \n
                        \n
                        "value_1": (float) ..., \n
                        ..., \n
                        "value_n": (float) ... \n
                    }
        """
        return_cellboxes = []
        for cellbox in self.cellboxes:
            if isinstance(cellbox, CellBox): #checking to filter out the cellbox that was splitted and replaced

                # Get json for CellBox
                #cell = cellbox.to_json()
                # Append ID to CellBox
                #cell['id'] = str(self.cellboxes.index(cellbox))

                return_cellboxes.append(cellbox)
        return return_cellboxes

    def get_cellbox(self, long, lat):
        """
            Returns the CellBox which contains a point, given by parameters lat, long

            Args:
                long (long): longitude of a given point
                lat (float): latitude of given point

            Returns:
                cellbox (CellBox): the cellbox which contains the point given my parameters
                (long, lat)
        """
        selected_cell = []
        for cellbox in self.cellboxes:
            if isinstance(cellbox, CellBox):
                if cellbox.contains_point(lat, long):
                    selected_cell.append(cellbox)
        return selected_cell[0]

    def get_bounds(self): 
        return self.boundary
  
    def get_max_split_depth (self):
        return self.max_split_depth
    
