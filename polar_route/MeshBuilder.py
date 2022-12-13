"""
In this section we will discuss the usage of the Mesh
functionality of PolarRoute.

Example:
    An example of how to run this code can be executed by running the
    following in an ipython/Jupyter Notebook::

        from RoutePlanner import Mesh

        import json
        with open('./config.json', 'r') as f:
            config = json.load(f)

        mesh = Mesh(config)
"""

import logging
import math
import numpy as np
import matplotlib.pyplot as plt

import json

from matplotlib.patches import Polygon as MatplotPolygon
from polar_route.cellbox import CellBox
import polar_route.data_loaders as data_loader

from PolarRoute.polar_route.Boundary import Boundary
from PolarRoute.polar_route.Direction import Direction
from PolarRoute.polar_route.EnvironmentMesh import EnvironmentMesh
from PolarRoute.polar_route.Metadata import Metadata
from PolarRoute.polar_route.NeighbourGraph import NeighbourGraph
from PolarRoute.polar_route.mesh import Mesh

class MeshBuilder:
    """
        Attributes:
            cellboxes (list<(CellBox)>): A list of CellBox objects forming the Mesh

            neighbour_graph (dict): A graphical representation of the adjacency
                relationship between CellBoxes in the Mesh. The neighbour_graph is
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

    def __init__(self, config):
        """
            Constructs a Mesh from a given config file.

            Args:
                config (dict): config file which defines the attributes of the Mesh
                    to be constructed. config is of the form - \n
                    \n
                    {\n
                        "config": {\n
                            "Mesh_info":{\n
                                "Region": {\n
                                    "latMin": (real),\n
                                    "latMax": (real),\n
                                    "longMin": (real),\n
                                    "longMax": (real),\n
                                    "startTime": (string) 'YYYY-MM-DD',\n
                                    "endTime": (string) 'YYYY-MM-DD',\n
                                    "cellWidth": (real),\n
                                    "cellHeight" (real),\n
                                    "splitDepth" (int)\n
                                },\n
                                "Data_sources": [
                                    {
                                        "loader": (string)\n
                                        "params" (dict)\n
                                    },\n
                                    ...,\n
                                    {...}

                                ]\n,
                                "splitting_conditions": [
                                    {
                                        <value>: {
                                            "threshold": (float),\n
                                            "upperBound": (float),\n
                                            "lowerBound": (float) \n
                                        }
                                    },\n
                                    ...,\n
                                    {...}\n
                                ]
                            }\n
                        }\n
                    }\n

                j_grid (bool): True if the Mesh to be constructed should be of the same
                    format as the original Java CellGrid, to be used for regression testing.
        """
        self.config = config      
        long_min = config['Mesh_info']['Region']['longMin']
        long_max = config['Mesh_info']['Region']['longMax']
        lat_min = config['Mesh_info']['Region']['latMin']
        lat_max = config['Mesh_info']['Region']['latMax']
        start_time = config['Mesh_info']['Region']['startTime']
        end_time = config['Mesh_info']['Region']['endTime']
        lat_range = [lat_min, lat_max]
        long_range = [long_min , long_max]
        time_range = [start_time , end_time]
        bounds = Boundary ( lat_range , long_range , time_range)


   
        cell_width = config['Mesh_info']['Region']['cellWidth']
        cell_height = config['Mesh_info']['Region']['cellHeight']

        assert (long_max - long_min) % cell_width == 0, \
            f"""The defined longitude region <{long_min} :{long_max}>
            is not divisable by the initial cell width <{cell_width}>"""

        assert (lat_max - lat_min) % cell_height == 0, \
            f"""The defined longitude region <{lat_min} :{lat_max}>
            is not divisable by the initial cell width <{cell_height}>"""

      

        # TODO: we should be using the inheritance hierarchy to achieve this
        '''

        self._j_grid = j_grid
        if 'j_grid' in config['Mesh_info'].keys():
            logging.warning("We're using the legacy Java style cell grid")
            self._j_grid = True

        '''
       

        logging.info("Initialising mesh...")

        logging.debug("Initialise cellBoxes...")
        cell_bounds = None
        cellboxes =[]
        self.neighbour_graph = None
        for lat in np.arange(lat_min, lat_max, cell_height):
            for long in np.arange(long_min, long_max, cell_width):
                cell_lat_range = [lat, lat+cell_height]
                cell_long_range = [long , long+cell_width]
                cell_bounds = Boundary (cell_lat_range , cell_long_range , time_range)
                cell_id = len (cellboxes)
                cellbox = CellBox(cell_bounds , cell_id)
                cellboxes.append(cellbox)

        grid_width = (long_max - long_min) / self._cell_width
        grid_height = (lat_max - lat_min) / self._cell_height

        ###########################################
        logging.debug("Initialise neighbours graph...")
        self.initialise_neoghbour_graph(cellboxes , grid_width) 
        ##########################################

        logging.debug("creating data_loaders...")
        min_datapoints = self.config['Mesh_info']['splitting']['minimum_datapoints']
        meta_data_list = []
        splitting_conds = []
        if 'Data_sources' in self.config['Mesh_info'].keys():
         for data_source in   self.config['Mesh_info']['Data_sources']:  
            loader_name = data_source['loader']
            # loader = DataLoaderFactory.get_data_loader( loader_name, data_source['params'] , min_datapoints)
            loader = None # to uncomment the previous line and use instead after itegrating wz Harry
            logging.debug("creating data loader {}".format(data_source['loader']))
           
            for split_cond in loader['splitting_conditions']:
                agg_type = loader["value_output_types"]
                value_fill_type = loader['value_fill_types']
                if (agg_type == ""):
                   agg_type = "MEAN"
                splitting_conds.append(split_cond)

            meta_data_obj = Metadata ( loader, splitting_conds , agg_type , value_fill_type)
            meta_data_list.append(meta_data_obj)
    
            

        for cellbox in cellboxes: # checking to avoid any dummy cellboxes (the ones that was splitted and replaced)
            if isinstance(cellbox, CellBox):
                cellbox.set_minimum_datapoints(min_datapoints)
                # assign meta data to each cellbox
                cellbox.set_data_source (meta_data_list)           
    ####################### 
        max_split_depth = self.config['Mesh_info']['splitting']['split_depth']
        self.mesh = Mesh(bounds , cellboxes , self.neighbour_graph, max_split_depth)
        self.mesh.set_config (config)

 ###############################       

    def to_json(self):
        """
            Returns this Mesh converted to a JSON object.

            Returns:
                json (json): a string representation of the CellGird parseable as a
                    JSON object. The JSON object is of the form -

                    {
                        "config": the config used to initialize the Mesh,
                        "cellboxes": a list of CellBoxes contained within the Mesh,
                        "neighbour_graph": a graph representing the adjacency of CellBoxes
                            within the Mesh
                    }
        """
        output = dict()
        output['config'] = self.config
        output["cellboxes"] = self.get_cellboxes()
        output['neighbour_graph'] = self.neighbour_graph


##############################
    def initialise_neighbour_graph (self , cellboxes ,grid_width):
        self.neighbour_graph = NeighbourGraph ()
        for cellbox in cellboxes:
            cellbox_indx = cellboxes.index(cellbox)
            neighbour_map = {1: [], 2: [], 3: [], 4: [], -1: [], -2: [], -3: [], -4: []}

            # add east neighbours to neighbour graph
            if (cellbox_indx + 1) % grid_width != 0:
                neighbour_map[2].append(cellbox_indx + 1)
                # south-east neighbours
                if cellbox_indx + grid_width < len(cellboxes):
                    neighbour_map[1].append(int((cellbox_indx + grid_width) + 1))
                # north-east neighbours
                if cellbox_indx - grid_width >= 0:
                    neighbour_map[3].append(int((cellbox_indx - grid_width) + 1))

            # add west neighbours to neighbour graph
            if cellbox_indx % grid_width != 0:
                neighbour_map[-2].append(cellbox_indx - 1)
                # add south-west neighbours to neighbour graph
                if cellbox_indx + grid_width < len(cellboxes):
                    neighbour_map[-3].append(int((cellbox_indx + grid_width) - 1))
                # add north-west neighbours to neighbour graph
                if cellbox_indx - grid_width >= 0:
                    neighbour_map[-1].append(int((cellbox_indx - grid_width) - 1))

            # add south neighbours to neighbour graph
            if cellbox_indx + grid_width < len(cellboxes):
                neighbour_map[-4].append(int(cellbox_indx + grid_width))

            # add north neighbours to neighbour graph
            if cellbox_indx - grid_width >= 0:
                neighbour_map[4].append(int(cellbox_indx - grid_width))

            self.neighbour_graph.add_node (cellbox_indx , neighbour_map)
      


    # Functions for splitting cellboxes within the Mesh

    def split_and_replace(self, cellbox):
        """
            Replaces a cellBox given by parameter 'cellBox' in this grid with
            4 smaller cellBoxes representing the four corners of the given cellBox.
            A neighbours map is then created for each of the 4 new cellBoxes
            and the neighbours map for all surrounding cell boxes is updated.

            Args:
                cellbox (CellBox): the CellBox within this Mesh to be split into
                    4 smaller CellBox objects.

        """
        split_cellboxes = cellbox.split()
        self.mesh.get_cellboxes().extend (split_cellboxes)
        cellboxes = self.mesh.get_cellboxes()
        cellbox_indx = cellboxes.index(cellbox)

        north_west_indx = cellboxes.index(split_cellboxes[0])
        north_east_indx = cellboxes.index(split_cellboxes[1])
        south_west_indx = cellboxes.index(split_cellboxes[2])
        south_east_indx = cellboxes.index(split_cellboxes[3])

        south_neighbour_indx = self.neighbour_graph.get_neighbours(cellbox_indx , 4)
        north_neighbour_indx = self.neighbour_graph.get_neighbours(cellbox_indx ,-4)
        east_neighbour_indx = self.neighbour_graph.get_neighbours(cellbox_indx ,2)
        west_neighbour_indx = self.neighbour_graph.get_neighbours(cellbox_indx ,-2)

        # Create neighbour map for SW split cell.
        sw_neighbour_map = {1: [north_east_indx],
                            2: [south_east_indx],
                            3: [],
                            4: [],
                            -1: self.neighbour_graph.get_neighbours(cellbox_indx ,-1), # update to use the NG class
                            -2: [],
                            -3: [],
                            -4: [north_west_indx]}

        self.fill_sw_neighbour_map(south_west_indx, south_neighbour_indx, west_neighbour_indx, sw_neighbour_map)
        self.neighbour_graph.add_node (south_west_indx ,sw_neighbour_map)

        # Create neighbour map for NW split cell
        nw_neighbour_map = {1: [],
                            2: [north_east_indx],
                            3: [south_east_indx],
                            4: [south_west_indx],
                            -1: [],
                            -2: [],
                            -3: self.neighbour_graph.get_neighbours(cellbox_indx ,-3),
                            -4: []}

        self.fill_nw_map( north_west_indx, north_neighbour_indx, west_neighbour_indx, nw_neighbour_map)
        self.neighbour_graph.add_node (north_west_indx , nw_neighbour_map)

        # Create neighbour map for NE split cell
        ne_neighbour_map = {1: self.neighbour_graph.get_neighbours(cellbox_indx ,1),
                            2: [],
                            3: [],
                            4: [south_east_indx],
                            -1: [south_west_indx],
                            -2: [north_west_indx],
                            -3: [],
                            -4: []}
        self.fill_ne_map( north_east_indx, north_neighbour_indx, east_neighbour_indx, ne_neighbour_map)
        self.neighbour_graph.add_node(north_east_indx, ne_neighbour_map)

        # Create neighbour map for SE split cell
        se_neighbour_map = {1: [],
                            2: [],
                            3: self.neighbour_graph.get_neighbours(cellbox_indx ,3),
                            4: [],
                            -1: [],
                            -2: [south_west_indx],
                            -3: [north_west_indx],
                            -4: [north_east_indx]}

        self.fill_se_map(south_east_indx, south_neighbour_indx, east_neighbour_indx, se_neighbour_map)
        self.neighbour_graph.add_node(south_east_indx , se_neighbour_map)

        # Update neighbours of cellbox_indx with the new neighbours coming from splitting the cellbox.
       
        self.neighbour_graph.update_neighbours (cellbox_indx, [north_west_indx, north_east_indx], Direction.north, cellboxes)  # Update north neighbours of cellbox_indx with the new neighbours comming from the splitted cellbox    
        self.neighbour_graph.update_neighbours (cellbox_indx, [north_east_indx, south_east_indx], Direction.east, cellboxes)   # Update east neighbour cellbox_indx with the new neighbours comming from the splitted cellbox 
        self.neighbour_graph.update_neighbours (cellbox_indx, [ south_west_indx, south_east_indx], Direction.south, cellboxes) # Update south neighbour cellbox_indx with the new neighbours comming from the splitted cellbox 
        self.neighbour_graph.update_neighbours (cellbox_indx, [ north_west_indx, south_west_indx], Direction.west, cellboxes)  # Update west neighbour cellbox_indx with the new neighbours comming from the splitted cellbox 
        # Update corner neighbour maps
        self.update_corner_neighbours(cellbox_indx, north_west_indx, north_east_indx, south_west_indx, south_east_indx)

        self.neighbour_graph.remove_node () #remove the original splitted cellbox from the neighbour_graph
        cellboxes[cellbox_indx] = None #set the original splitted cellbox to its None 
      



 ############################## methods to fill the neighbour maps of the splitted cells ########################
    # method that fills the South east neighbours 
    def fill_se_map(self, south_east_indx, south_neighbour_indx, east_neighbour_indx, se_neighbour_map):
        cellboxes = self.mesh.get_cellboxes()
        for indx in south_neighbour_indx:
            if self.neighbour_graph.get_neighbour_case(cellboxes[south_east_indx], cellboxes[indx]) == 4:
                se_neighbour_map[4].append(indx)
            if self.neighbour_graph.get_neighbour_case(cellboxes[south_east_indx], cellboxes[indx]) == -1:
                se_neighbour_map[-1].append(indx)
        for indx in east_neighbour_indx:
            if self.neighbour_graph.get_neighbour_case(cellboxes[south_east_indx], cellboxes[indx]) == 2:
                se_neighbour_map[2].append(indx)
            if self.neighbour_graph.get_neighbour_case(cellboxes[south_east_indx], cellboxes[indx]) == 1:
                se_neighbour_map[1].append(indx)

    # method that fills the North east neighbours  
    def fill_ne_map(self, north_east_indx, north_neighbour_indx, east_neighbour_indx, ne_neighbour_map):
        cellboxes = self.mesh.get_cellboxes()
        for indx in north_neighbour_indx:
            if self.neighbour_graph.get_neighbour_case(cellboxes[north_east_indx], cellboxes[indx]) == -4:
                ne_neighbour_map[-4].append(indx)
            if self.neighbour_graph.get_neighbour_case(cellboxes[north_east_indx], cellboxes[indx]) == -3:
                ne_neighbour_map[-3].append(indx)
        for indx in east_neighbour_indx:
            if self.neighbour_graph.get_neighbour_case(cellboxes[north_east_indx], cellboxes[indx]) == 2:
                ne_neighbour_map[2].append(indx)
            if self.neighbour_graph.get_neighbour_case(cellboxes[north_east_indx], cellboxes[indx]) == 3:
                ne_neighbour_map[3].append(indx)

    # method that fills the North west neighbours 
    def fill_nw_map(self,  north_west_indx, north_neighbour_indx, west_neighbour_indx, nw_neighbour_map):
        cellboxes = self.mesh.get_cellboxes()
        for indx in north_neighbour_indx:
            if self.neighbour_graph.get_neighbour_case(cellboxes[north_west_indx], cellboxes[indx]) == -4:
                nw_neighbour_map[-4].append(indx)
            if self.neighbour_graph.get_neighbour_case(cellboxes[north_west_indx], cellboxes[indx]) == 1:
                nw_neighbour_map[1].append(indx)
        for indx in west_neighbour_indx:
            if self.neighbour_graph.get_neighbour_case(cellboxes[north_west_indx], cellboxes[indx]) == -2:
                nw_neighbour_map[-2].append(indx)
            if self.neighbour_graph.get_neighbour_case(cellboxes[north_west_indx], cellboxes[indx]) == -1:
                nw_neighbour_map[-1].append(indx)
    
    # method that fills the South west neighbours 
    def fill_sw_neighbour_map(self, south_west_indx, south_neighbour_indx, west_neighbour_indx, sw_neighbour_map):
        cellboxes = self.mesh.get_cellboxes()
        for indx in south_neighbour_indx:
            if self.neighbour_graph.get_neighbour_case(cellboxes[south_west_indx], cellboxes[indx]) == 3:
                sw_neighbour_map[3].append(indx)
            if self.neighbour_graph.get_neighbour_case(cellboxes[south_west_indx], cellboxes[indx]) == 4:
                sw_neighbour_map[4].append(indx)
        for indx in west_neighbour_indx:
            if self.neighbour_graph.get_neighbour_case(cellboxes[south_west_indx], cellboxes[indx]) == -2:
                sw_neighbour_map[-2].append(indx)
            if self.neighbour_graph.get_neighbour_case(cellboxes[south_west_indx], cellboxes[indx]) == -3:
                sw_neighbour_map[-3].append(indx)

###################################################################################################
    def split_to_depth(self, split_depth):
        """
            splits all cellboxes in this grid until a maximum split depth
            is reached, or all cellboxes are homogeneous.

            Args:
                split_depth (int): The maximum split depth reached by any CellBox
                    within this Mesh after splitting.
        """
        for cellbox in self.mesh.get_cellboxes():
            if isinstance(cellbox, CellBox):
                if (cellbox.split_depth < split_depth) & (cellbox.should_split()):
                    self.split_and_replace(cellbox) 
#################################################################################################
    def build_environmental_mesh(self):
        """
            splits the mesh then goes through the mesh cellboxes and builds an evironmental mesh that contains the cellboxes aggregated data

        """
        self.split_to_depth(self.mesh.get_max_split_depth())
        agg_cellboxes = []
        for cellbox in self.mesh.get_cellboxes():
            if isinstance(cellbox, CellBox):
               agg_cellboxes.append (cellbox.aggregate()) 
        env_mesh = EnvironmentMesh (self.mesh.get_bounds() , agg_cellboxes , self.neighbour_graph ,self.get_config())
        return env_mesh

#################################################################################################
    @property
    def get_config(self):
        return self._config
