"""

In this section we will discuss the usage of the MeshBuilder functionality of PolarRoute.\n

Example:
    An example of how to run this code can be executed by running the following in an ipython/Jupyter Notebook::\n

        from polar_route.mesh_generation.mesh_builder import MeshBuilder

        import json
        with open('./config.json', 'r') as f:
            config = json.load(f)

        mesh_builder = MeshBuilder(config)
        mesh_builder.build_environmental_mesh() \n

"""

import logging
import math
import json
import numpy as np
from polar_route.mesh_generation.jgrid_cellbox import JGridCellBox
from polar_route.mesh_generation.boundary import Boundary
from polar_route.mesh_generation.cellbox import CellBox
from polar_route.mesh_generation.direction import Direction
from polar_route.mesh_generation.environment_mesh import EnvironmentMesh
from polar_route.mesh_generation.metadata import Metadata
from polar_route.mesh_generation.neighbour_graph import NeighbourGraph
from polar_route.mesh_generation.mesh import Mesh
from polar_route.Dataloaders.Factory import DataLoaderFactory


class MeshBuilder:
    """

       A class resposible for building an environment mesh based on a provided config file.
       
    """

    def __init__(self, config):
        """

            Constructs a Mesh from a given config file.\n

            Args:
                config (dict): config file which defines the attributes of the Mesh to be constructed. config is of the form: \n
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
                                "Data_sources": [\n
                                    {\n
                                        "loader": (string)\n
                                        "params" (dict)\n
                                    }\n
                                ],\n
                                "splitting": { \n
                                    "split_depth": (int),\n
                                    "minimum_datapoints": (int)\n
                                    }\n
                            }\n
                        }\n
                   
                    

                    "j_grid" (bool): True if the Mesh to be constructed should be of the same format as the original Java CellGrid, to be used for regression testing.\n
                
         """
        self.config = config
        bounds = Boundary.from_json(config)

        cell_width = config['Mesh_info']['Region']['cellWidth']
        cell_height = config['Mesh_info']['Region']['cellHeight']

        self.validate_bounds(bounds, cell_width, cell_height)

        logging.info("Initialising mesh...")
        logging.info("Initialising cellboxes...")
     

        logging.debug("Initialising cellBoxes...")
        cellboxes = []
        cellboxes = self.initialize_cellboxes(bounds, cell_width, cell_height)

        grid_width = (bounds.get_long_max() -
                      bounds.get_long_min()) / cell_width

  

        min_datapoints = 5
        if 'splitting' in self.config['Mesh_info']:
            min_datapoints = self.config['Mesh_info']['splitting']['minimum_datapoints']
        meta_data_list = self.initialize_meta_data(bounds, min_datapoints)

        # checking to avoid any dummy cellboxes (the ones that was splitted and replaced)
        logging.info("Assigning data source to cellboxes...")
        for cellbox in cellboxes:
            if isinstance(cellbox, CellBox):
                cellbox.set_minimum_datapoints(min_datapoints)
                # assign meta data to each cellbox
                cellbox.set_data_source(meta_data_list)

        
        logging.info("Initialise neighbours graph...")
        self.neighbour_graph = NeighbourGraph(cellboxes, grid_width)

        max_split_depth = 0
        if 'splitting' in self.config['Mesh_info']:
            max_split_depth = self.config['Mesh_info']['splitting']['split_depth']
        self.mesh = Mesh(bounds, cellboxes,
                         self.neighbour_graph, max_split_depth)
        self.mesh.set_config(config)

    def initialize_meta_data(self, bounds, min_datapoints):
        meta_data_list = []
        splitting_conds = []
        if 'Data_sources' in self.config['Mesh_info'].keys():
            for data_source in self.config['Mesh_info']['Data_sources']:
                loader_name = data_source['loader']
                loader = DataLoaderFactory().get_dataloader(
                    loader_name, bounds, data_source['params'], min_datapoints)

                logging.debug("creating data loader {}".format(
                    data_source['loader']))
                updated_splitiing_cond = []  # create this list to get rid of the data_name in the conditions as it is not handeled by the DataLoader, remove after talking to Harry to address this in the loader
                if 'splitting_conditions' in data_source['params']:
                    splitting_conds = data_source['params']['splitting_conditions']
                    for split_cond in splitting_conds:
                        cond = split_cond[loader.data_name]
                        updated_splitiing_cond.append(cond)

                value_fill_type = data_source['params']['value_fill_types']

                meta_data_obj = Metadata(
                    loader, updated_splitiing_cond,  value_fill_type)
                meta_data_list.append(meta_data_obj)

        return meta_data_list

    def is_jgrid_mesh(self):
        if 'j_grid' in self.config['Mesh_info'].keys():
            logging.warning("We're using the legacy Java style cell grid")
            return True
        return False

    def initialize_cellboxes(self, bounds, cell_width, cell_height):
        cellboxes = []
        grid_width = (bounds.get_long_max() -
                      bounds.get_long_min()) / cell_width
        grid_height = (bounds.get_lat_max() -
                       bounds.get_lat_min()) / cell_height
        for lat in np.arange(bounds.get_lat_min(), bounds.get_lat_max(), cell_height):
            for long in np.arange(bounds.get_long_min(), bounds.get_long_max(), cell_width):
                cell_lat_range = [lat, lat+cell_height]
                cell_long_range = [long, long+cell_width]
                cell_bounds = Boundary(
                    cell_lat_range, cell_long_range, bounds.get_time_range())
                cell_id = str(len(cellboxes))
                if self.is_jgrid_mesh():
                    cellbox_indx = len(cellboxes)
                    cellbox = JGridCellBox(cell_bounds, cell_id)
                    x_coord = cellbox_indx % grid_width
                    y_coord = abs(math.floor(
                        cellbox_indx / grid_width) - (grid_height - 1))
                    cellbox.set_grid_coord(x_coord, y_coord)
                    cellbox.set_initial_bounds(bounds)

                else:
                    cellbox = CellBox(cell_bounds, cell_id)
                cellboxes.append(cellbox)
        return cellboxes

    def validate_bounds(self, bounds, cell_width, cell_height):
        assert (bounds.get_long_max() - bounds.get_long_min()) % cell_width == 0, \
            f"""The defined longitude region <{bounds.get_long_min()} :{bounds.get_long_max()}>
            is not divisable by the initial cell width <{cell_width}>"""

        assert (bounds.get_lat_max() - bounds.get_lat_min()) % cell_height == 0, \
            f"""The defined longitude region <{bounds.get_lat_min()} :{bounds.get_lat_max()}>
            is not divisable by the initial cell width <{cell_height}>"""

    def to_json(self):
        """
            Returns this Mesh converted to a JSON object.

            Returns:
                json (json): a string representation of the CellGird parseable as a JSON object. The JSON object is of the form -

                    {
                        "config": the config used to initialize the Mesh,
                        "cellboxes": a list of CellBoxes contained within the Mesh,
                        "neighbour_graph": a graph representing the adjacency of CellBoxes within the Mesh
                    }
        """
        output = dict()
        output['config'] = self.config
        output["cellboxes"] = self.mesh.get_cellboxes()
        output['neighbour_graph'] = self.neighbour_graph.get_graph()


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
        split_cellboxes = cellbox.split(len(self.mesh.cellboxes))
        self.mesh.cellboxes += split_cellboxes
        cellboxes = self.mesh.cellboxes
        cellbox_indx = cellboxes.index(cellbox)

        north_west_indx = cellboxes.index(split_cellboxes[0])
        north_east_indx = cellboxes.index(split_cellboxes[1])
        south_west_indx = cellboxes.index(split_cellboxes[2])
        south_east_indx = cellboxes.index(split_cellboxes[3])

        south_neighbour_indx = self.neighbour_graph.get_neighbours(
            cellbox_indx, 4)
        north_neighbour_indx = self.neighbour_graph.get_neighbours(
            cellbox_indx, -4)
        east_neighbour_indx = self.neighbour_graph.get_neighbours(
            cellbox_indx, 2)
        west_neighbour_indx = self.neighbour_graph.get_neighbours(
            cellbox_indx, -2)

        # Create neighbour map for SW split cell.
        sw_neighbour_map = {1: [north_east_indx],
                            2: [south_east_indx],
                            3: [],
                            4: [],
                            # update to use the NG class
                            -1: self.neighbour_graph.get_neighbours(cellbox_indx, -1),
                            -2: [],
                            -3: [],
                            -4: [north_west_indx]}

        self.fill_sw_neighbour_map(
            south_west_indx, south_neighbour_indx, west_neighbour_indx, sw_neighbour_map)
        self.neighbour_graph.add_node(south_west_indx, sw_neighbour_map)

        # Create neighbour map for NW split cell
        nw_neighbour_map = {1: [],
                            2: [north_east_indx],
                            3: [south_east_indx],
                            4: [south_west_indx],
                            -1: [],
                            -2: [],
                            -3: self.neighbour_graph.get_neighbours(cellbox_indx, -3),
                            -4: []}

        self.fill_nw_map(north_west_indx, north_neighbour_indx,
                         west_neighbour_indx, nw_neighbour_map)
        self.neighbour_graph.add_node(north_west_indx, nw_neighbour_map)

        # Create neighbour map for NE split cell
        ne_neighbour_map = {1: self.neighbour_graph.get_neighbours(cellbox_indx, 1),
                            2: [],
                            3: [],
                            4: [south_east_indx],
                            -1: [south_west_indx],
                            -2: [north_west_indx],
                            -3: [],
                            -4: []}
        self.fill_ne_map(north_east_indx, north_neighbour_indx,
                         east_neighbour_indx, ne_neighbour_map)
        self.neighbour_graph.add_node(north_east_indx, ne_neighbour_map)

        # Create neighbour map for SE split cell
        se_neighbour_map = {1: [],
                            2: [],
                            3: self.neighbour_graph.get_neighbours(cellbox_indx, 3),
                            4: [],
                            -1: [],
                            -2: [south_west_indx],
                            -3: [north_west_indx],
                            -4: [north_east_indx]}

        self.fill_se_map(south_east_indx, south_neighbour_indx,
                         east_neighbour_indx, se_neighbour_map)
        self.neighbour_graph.add_node(south_east_indx, se_neighbour_map)

        # Update neighbours of cellbox_indx with the new neighbours coming from splitting the cellbox.

        # Update north neighbours of cellbox_indx with the new neighbours comming from the splitted cellbox
        self.neighbour_graph.update_neighbours(
            cellbox_indx, [north_west_indx, north_east_indx], Direction.north, cellboxes)
        # Update east neighbour cellbox_indx with the new neighbours comming from the splitted cellbox
        self.neighbour_graph.update_neighbours(
            cellbox_indx, [north_east_indx, south_east_indx], Direction.east, cellboxes)
        # Update south neighbour cellbox_indx with the new neighbours comming from the splitted cellbox
        self.neighbour_graph.update_neighbours(
            cellbox_indx, [south_west_indx, south_east_indx], Direction.south, cellboxes)
        # Update west neighbour cellbox_indx with the new neighbours comming from the splitted cellbox
        self.neighbour_graph.update_neighbours(
            cellbox_indx, [north_west_indx, south_west_indx], Direction.west, cellboxes)
        # Update corner neighbour maps
        self.neighbour_graph.update_corner_neighbours(
            cellbox_indx, north_west_indx, north_east_indx, south_west_indx, south_east_indx)

        # remove the original splitted cellbox from the neighbour_graph
        self.neighbour_graph.remove_node(cellbox_indx)
        # set the original splitted cellbox to None
        cellboxes[cellbox_indx] = None

 ############################## methods to fill the neighbour maps of the splitted cells ########################
 
    def fill_se_map(self, south_east_indx, south_neighbour_indx, east_neighbour_indx, se_neighbour_map):
        """
             method that fills the South east neighbours
        """
        cellboxes = self.mesh.cellboxes
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

   
    def fill_ne_map(self, north_east_indx, north_neighbour_indx, east_neighbour_indx, ne_neighbour_map):
        """
             method that fills the North east neighbours
        """
        cellboxes = self.mesh.cellboxes
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

    
    def fill_nw_map(self,  north_west_indx, north_neighbour_indx, west_neighbour_indx, nw_neighbour_map):
        """
             method that fills the North west neighbours
        """
        cellboxes = self.mesh.cellboxes
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

  
    def fill_sw_neighbour_map(self, south_west_indx, south_neighbour_indx, west_neighbour_indx, sw_neighbour_map):
        """
             method that fills the South west neighbours
        """
        cellboxes = self.mesh.cellboxes
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


    def split_to_depth(self, split_depth):
        """
            splits all cellboxes in this grid until a maximum split depth
            is reached, or all cellboxes are homogeneous.

            Args:
                split_depth (int): The maximum split depth reached by any CellBox
                    within this Mesh after splitting.
        """
        logging.info ("splitting cellboxes ...")
        # loop over the data_sources then cellboxes to implement depth-first splitting. should be simpler and loop over cellboxes only once we switch to breadth-first splitting
        # this impl assumws all the cellboxes have the same data sources. should not be the caase once we switch to breadth-first splitting.
        data_sources = self.mesh.cellboxes[0].get_data_source()
        for index in range(0, len(data_sources)):
            if (len(data_sources[index].get_splitting_conditions()) > 0):
                for cellbox in self.mesh.cellboxes:
                    if isinstance(cellbox, CellBox):
                        should_split = cellbox.should_split(index+1)
                        if (cellbox.get_split_depth() < split_depth) & should_split:
                            self.split_and_replace(cellbox)

    def build_environmental_mesh(self):
        """
            splits the mesh then goes through the mesh cellboxes and builds an evironmental mesh that contains the cellboxes aggregated data
            
            Returns:
                EnvironmentMesh: an object that represents the constructed nonunifrom mesh and contains the aggregated cellboxs and neighbour graph 
        """
        self.split_to_depth(self.mesh.get_max_split_depth())
        agg_cellboxes = []
        for cellbox in self.mesh.cellboxes:
            if isinstance(cellbox, CellBox):
                agg_cellboxes.append(cellbox.aggregate())

        env_mesh = EnvironmentMesh(self.mesh.get_bounds(
        ), agg_cellboxes, self.neighbour_graph, self.get_config())

        return env_mesh

    def get_config(self):
        """
        returns the config
        """
        return self.config


if __name__ == '__main__':
    import time
    start = time.time()
    conf = None
    with open("../feb_2013_Jgrid_config.json", "r") as config_file:
        conf = json.load(config_file)['config']

    mesh_builder = MeshBuilder(conf)
    env_mesh = mesh_builder.build_environmental_mesh()
    print(conf)
    # with open("feb_2013_Jgrid_.json", 'w') as file:
    #     json.dump(env_mesh.to_json(), file)
    env_mesh.save ("feb_2013_Jgrid_.json")
    end = time.time()
    elapsed_seconds = float("%.2f" % (end - start))
    print(elapsed_seconds)
