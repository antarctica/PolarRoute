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

# FIXME: This is weird
import json as JSON

from matplotlib.patches import Polygon as MatplotPolygon
from polar_route.cellbox import CellBox
import polar_route.data_loaders as data_loader


class Mesh:
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

    def __init__(self, config, j_grid=False):
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
        self._config = config

        self._long_min = config['Mesh_info']['Region']['longMin']
        self._long_max = config['Mesh_info']['Region']['longMax']
        self._lat_min = config['Mesh_info']['Region']['latMin']
        self._lat_max = config['Mesh_info']['Region']['latMax']

        self._cell_width = config['Mesh_info']['Region']['cellWidth']
        self._cell_height = config['Mesh_info']['Region']['cellHeight']

        assert (self._long_max - self._long_min) % self._cell_width == 0, \
            f"""The defined longitude region <{self._long_min} :{self._long_max}>
            is not divisable by the initial cell width <{self._cell_width}>"""

        assert (self._lat_max - self._lat_min) % self._cell_height == 0, \
            f"""The defined longitude region <{self._lat_min} :{self._lat_max}>
            is not divisable by the initial cell width <{self._cell_height}>"""

        self._start_time = config['Mesh_info']['Region']['startTime']
        self._end_time = config['Mesh_info']['Region']['endTime']

        # TODO: we should be using the inheritance hierarchy to achieve this
        self._j_grid = j_grid
        if 'j_grid' in config['Mesh_info'].keys():
            logging.warning("We're using the legacy Java style cell grid")
            self._j_grid = True

        # TODO: getters / setters (look at config)
        self.cellboxes = []
        self.neighbour_graph = {}
#        self.__init()


#    def __init(self):
        logging.info("Initialising mesh...")

        logging.debug("Initialise cellBoxes")
        for lat in np.arange(self._lat_min, self._lat_max, self._cell_height):
            for long in np.arange(self._long_min, self._long_max, self._cell_width):
                cellbox = CellBox(lat, long, self._cell_width, self._cell_height,
                                  splitting_conditions=[], j_grid=self._j_grid)
                self.cellboxes.append(cellbox)

        grid_width = (self._long_max - self._long_min) / self._cell_width
        grid_height = (self._lat_max - self._lat_min) / self._cell_height

        logging.debug("Calculate initial neighbours graph")
        for cellbox in self.cellboxes:
            cellbox_indx = self.cellboxes.index(cellbox)
            neighbour_map = {1: [], 2: [], 3: [], 4: [], -1: [], -2: [], -3: [], -4: []}

            # add east neighbours to neighbour graph
            if (cellbox_indx + 1) % grid_width != 0:
                neighbour_map[2].append(cellbox_indx + 1)
                # south-east neighbours
                if cellbox_indx + grid_width < len(self.cellboxes):
                    neighbour_map[1].append(int((cellbox_indx + grid_width) + 1))
                # north-east neighbours
                if cellbox_indx - grid_width >= 0:
                    neighbour_map[3].append(int((cellbox_indx - grid_width) + 1))

            # add west neighbours to neighbour graph
            if cellbox_indx % grid_width != 0:
                neighbour_map[-2].append(cellbox_indx - 1)
                # add south-west neighbours to neighbour graph
                if cellbox_indx + grid_width < len(self.cellboxes):
                    neighbour_map[-3].append(int((cellbox_indx + grid_width) - 1))
                # add north-west neighbours to neighbour graph
                if cellbox_indx - grid_width >= 0:
                    neighbour_map[-1].append(int((cellbox_indx - grid_width) - 1))

            # add south neighbours to neighbour graph
            if cellbox_indx + grid_width < len(self.cellboxes):
                neighbour_map[-4].append(int(cellbox_indx + grid_width))

            # add north neighbours to neighbour graph
            if cellbox_indx - grid_width >= 0:
                neighbour_map[4].append(int(cellbox_indx - grid_width))

            self.neighbour_graph[cellbox_indx] = neighbour_map

            # set value output types of a cellbox
            if 'value_output_types' in self.config['Mesh_info'].keys():
                cellbox.add_value_output_type(self.config['Mesh_info']['value_output_types'])

            # set gridCoord of cellBox
            x_coord = cellbox_indx % grid_width
            y_coord = abs(math.floor(cellbox_indx / grid_width) - (grid_height - 1))
            cellbox.set_grid_coord(x_coord, y_coord)

            # set focus of cellBox
            cellbox.set_focus([])

        logging.debug("calling data_loaders:")
        # j_grids represent currents differently from other data sources, so a bispoke
        # function 'add_current_points' must be called
        if self._j_grid:
            loader_name = self.config['Mesh_info']['j_grid']['Currents']['loader']
            loader = getattr(data_loader, loader_name)
            logging.debug("J_grid using loader {}".format(loader_name))

            data_points = loader(self.config['Mesh_info']['j_grid']['Currents']['params'],
                self._long_min, self._long_max, self._lat_min, self._lat_max,
                self._start_time, self._end_time)

            self.add_current_points(data_points)

        if 'Data_sources' in self.config['Mesh_info'].keys():
            for data_source in self.config['Mesh_info']['Data_sources']:
                loader = getattr(data_loader, data_source['loader'])
                logging.debug("Using data loader {}".format(data_source['loader']))

                data_points = loader(data_source['params'],
                                     self._long_min, self._long_max, self._lat_min, self._lat_max,
                                     self._start_time, self._end_time)

                self.add_data_points(data_points)

        logging.debug("Add splitting conditions from config and split Mesh.")
        self.splitting_conditions = []
        if 'splitting' in self.config['Mesh_info'].keys():
            for splitting_condition in self.config['Mesh_info']['splitting']['splitting_conditions']:
                logging.debug("Adding condition on {} to all cellboxes".format(splitting_condition))
                logging.debug("Number of cellboxes before splitting: {}".format(len(self.cellboxes)))
                for cellbox in self.cellboxes:
                    if isinstance(cellbox, CellBox):
                        cellbox.add_splitting_condition(splitting_condition)
                        cellbox.set_minimum_datapoints(self.config['Mesh_info']['splitting']['minimum_datapoints'])
                        cellbox.set_value_fill_types(self.config['Mesh_info']['splitting']['value_fill_types'])

                logging.debug("Splitting to depth {}".format(self.config['Mesh_info']['splitting']['split_depth']))
                self.split_to_depth(self.config['Mesh_info']['splitting']['split_depth'])
                logging.debug("Number of cellboxes after splitting: {}".format(len(self.cellboxes)))

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
            if isinstance(cellbox, CellBox):

                # Get json for CellBox
                cell = cellbox.to_json()
                # Append ID to CellBox
                cell['id'] = str(self.cellboxes.index(cellbox))

                return_cellboxes.append(cell)
        return return_cellboxes

    def to_json(self):
        """
            Returns this Mesh converted to string parsable as a JSON object.

            Returns:
                json (string): a string representation of the CellGird parseable as a
                    JSON object. The JSON object is of the form -

                    {
                        "config": the config used to initialize the Mesh,
                        "cellboxes": a list of CellBoxes contained within the Mesh,
                        "neighbour_graph": a graph representing the adjacency of CellBoxes
                            within the Mesh
                    }
        """
        json = dict()
        json['config'] = self.config
        json["cellboxes"] = self.get_cellboxes()
        json['neighbour_graph'] = self.neighbour_graph

        # FIXME: Eh?
        return JSON.loads(JSON.dumps(json))

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

        self.cellboxes += split_cellboxes

        cellbox_indx = self.cellboxes.index(cellbox)

        north_west_indx = self.cellboxes.index(split_cellboxes[0])
        north_east_indx = self.cellboxes.index(split_cellboxes[1])
        south_west_indx = self.cellboxes.index(split_cellboxes[2])
        south_east_indx = self.cellboxes.index(split_cellboxes[3])

        south_neighbour_indx = self.neighbour_graph[cellbox_indx][4]
        north_neighbour_indx = self.neighbour_graph[cellbox_indx][-4]
        east_neighbour_indx = self.neighbour_graph[cellbox_indx][2]
        west_neighbour_indx = self.neighbour_graph[cellbox_indx][-2]

        # Create neighbour map for SW split cell.
        sw_neighbour_map = {1: [north_east_indx],
                            2: [south_east_indx],
                            3: [],
                            4: [],
                            -1: self.neighbour_graph[cellbox_indx][-1],
                            -2: [],
                            -3: [],
                            -4: [north_west_indx]}

        for indx in south_neighbour_indx:
            if self.get_neighbour_case(self.cellboxes[south_west_indx], self.cellboxes[indx]) == 3:
                sw_neighbour_map[3].append(indx)
            if self.get_neighbour_case(self.cellboxes[south_west_indx], self.cellboxes[indx]) == 4:
                sw_neighbour_map[4].append(indx)
        for indx in west_neighbour_indx:
            if self.get_neighbour_case(self.cellboxes[south_west_indx], self.cellboxes[indx]) == -2:
                sw_neighbour_map[-2].append(indx)
            if self.get_neighbour_case(self.cellboxes[south_west_indx], self.cellboxes[indx]) == -3:
                sw_neighbour_map[-3].append(indx)

        self.neighbour_graph[south_west_indx] = sw_neighbour_map

        # Create neighbour map for NW split cell
        nw_neighbour_map = {1: [],
                            2: [north_east_indx],
                            3: [south_east_indx],
                            4: [south_west_indx],
                            -1: [],
                            -2: [],
                            -3: self.neighbour_graph[cellbox_indx][-3],
                            -4: []}

        for indx in north_neighbour_indx:
            if self.get_neighbour_case(self.cellboxes[north_west_indx], self.cellboxes[indx]) == -4:
                nw_neighbour_map[-4].append(indx)
            if self.get_neighbour_case(self.cellboxes[north_west_indx], self.cellboxes[indx]) == 1:
                nw_neighbour_map[1].append(indx)
        for indx in west_neighbour_indx:
            if self.get_neighbour_case(self.cellboxes[north_west_indx], self.cellboxes[indx]) == -2:
                nw_neighbour_map[-2].append(indx)
            if self.get_neighbour_case(self.cellboxes[north_west_indx], self.cellboxes[indx]) == -1:
                nw_neighbour_map[-1].append(indx)

        self.neighbour_graph[north_west_indx] = nw_neighbour_map

        # Create neighbour map for NE split cell
        ne_neighbour_map = {1: self.neighbour_graph[cellbox_indx][1],
                            2: [],
                            3: [],
                            4: [south_east_indx],
                            -1: [south_west_indx],
                            -2: [north_west_indx],
                            -3: [],
                            -4: []}

        for indx in north_neighbour_indx:
            if self.get_neighbour_case(self.cellboxes[north_east_indx], self.cellboxes[indx]) == -4:
                ne_neighbour_map[-4].append(indx)
            if self.get_neighbour_case(self.cellboxes[north_east_indx], self.cellboxes[indx]) == -3:
                ne_neighbour_map[-3].append(indx)
        for indx in east_neighbour_indx:
            if self.get_neighbour_case(self.cellboxes[north_east_indx], self.cellboxes[indx]) == 2:
                ne_neighbour_map[2].append(indx)
            if self.get_neighbour_case(self.cellboxes[north_east_indx], self.cellboxes[indx]) == 3:
                ne_neighbour_map[3].append(indx)

        self.neighbour_graph[north_east_indx] = ne_neighbour_map

        # Create neighbour map for SE split cell
        se_neighbour_map = {1: [],
                            2: [],
                            3: self.neighbour_graph[cellbox_indx][3],
                            4: [],
                            -1: [],
                            -2: [south_west_indx],
                            -3: [north_west_indx],
                            -4: [north_east_indx]}

        for indx in south_neighbour_indx:
            if self.get_neighbour_case(self.cellboxes[south_east_indx], self.cellboxes[indx]) == 4:
                se_neighbour_map[4].append(indx)
            if self.get_neighbour_case(self.cellboxes[south_east_indx], self.cellboxes[indx]) == -1:
                se_neighbour_map[-1].append(indx)
        for indx in east_neighbour_indx:
            if self.get_neighbour_case(self.cellboxes[south_east_indx], self.cellboxes[indx]) == 2:
                se_neighbour_map[2].append(indx)
            if self.get_neighbour_case(self.cellboxes[south_east_indx], self.cellboxes[indx]) == 1:
                se_neighbour_map[1].append(indx)

        self.neighbour_graph[south_east_indx] = se_neighbour_map

        # Update neighbour map of neighbours of the split box.

        # Update north neighbour map
        for indx in north_neighbour_indx:
            self.neighbour_graph[indx][4].remove(cellbox_indx)

            crossing_case = self.get_neighbour_case(self.cellboxes[indx],
                                                    self.cellboxes[north_west_indx])
            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(north_west_indx)

            crossing_case = self.get_neighbour_case(self.cellboxes[indx],
                                                    self.cellboxes[north_east_indx])
            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(north_east_indx)

        # Update east neighbour map
        for indx in east_neighbour_indx:
            self.neighbour_graph[indx][-2].remove(cellbox_indx)

            crossing_case = self.get_neighbour_case(self.cellboxes[indx],
                                                    self.cellboxes[north_east_indx])
            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(north_east_indx)

            crossing_case = self.get_neighbour_case(self.cellboxes[indx],
                                                    self.cellboxes[south_east_indx])
            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(south_east_indx)

        # Update south neighbour map
        for indx in south_neighbour_indx:
            self.neighbour_graph[indx][-4].remove(cellbox_indx)

            crossing_case = self.get_neighbour_case(self.cellboxes[indx],
                                                    self.cellboxes[south_east_indx])
            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(south_east_indx)

            crossing_case = self.get_neighbour_case(self.cellboxes[indx],
                                                    self.cellboxes[south_west_indx])
            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(south_west_indx)

        # Update west neighbour map
        for indx in west_neighbour_indx:
            self.neighbour_graph[indx][2].remove(cellbox_indx)

            crossing_case = self.get_neighbour_case(self.cellboxes[indx],
                                                    self.cellboxes[north_west_indx])
            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(north_west_indx)

            crossing_case = self.get_neighbour_case(self.cellboxes[indx],
                                                    self.cellboxes[south_west_indx])
            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(south_west_indx)

        # Update corner neighbour maps
        north_east_corner_indx = self.neighbour_graph[cellbox_indx][1]
        if len(north_east_corner_indx) > 0:
            self.neighbour_graph[north_east_corner_indx[0]][-1] = [north_east_indx]

        north_west_corner_indx = self.neighbour_graph[cellbox_indx][-3]
        if len(north_west_corner_indx) > 0:
            self.neighbour_graph[north_west_corner_indx[0]][3] = [north_west_indx]

        south_east_corner_indx = self.neighbour_graph[cellbox_indx][3]
        if len(south_east_corner_indx) > 0:
            self.neighbour_graph[south_east_corner_indx[0]][-3] = [south_east_indx]

        south_west_corner_indx = self.neighbour_graph[cellbox_indx][-1]
        if len(south_west_corner_indx) > 0:
            self.neighbour_graph[south_west_corner_indx[0]][1] = [south_west_indx]

        split_container = {"northEast": north_east_indx,
                           "northWest": north_west_indx,
                           "southEast": south_east_indx,
                           "southWest": south_west_indx}

        self.cellboxes[cellbox_indx] = split_container
        self.neighbour_graph.pop(cellbox_indx)

    def split_to_depth(self, split_depth):
        """
            splits all cellboxes in this grid until a maximum split depth
            is reached, or all cellboxes are homogeneous.

            Args:
                split_depth (int): The maximum split depth reached by any CellBox
                    within this Mesh after splitting.
        """
        for cellbox in self.cellboxes:
            if isinstance(cellbox, CellBox):
                if (cellbox.split_depth < split_depth) & (cellbox.should_split()):
                    self.split_and_replace(cellbox)

    # Functions for debugging
    def plot(self, highlight_cellboxes={}, plot_ice=True, plot_currents=False,
             plot_borders=True, paths=None, routepoints=False, waypoints=None):
        """
            creates and displays a plot for this Mesh

            To be used for debugging purposes only.
        """
        # Create plot figure
        fig, axis = plt.subplots(1, 1, figsize=(25, 11))

        fig.patch.set_facecolor('white')
        axis.set_facecolor('lightblue')

        for cellbox in self.cellboxes:
            if isinstance(cellbox, CellBox):
                # plot ice
                if plot_ice and not np.isnan(cellbox.get_value('iceArea')):
                    if self._j_grid:
                        if cellbox.get_value('iceArea') >= 0.04:
                            axis.add_patch(MatplotPolygon(cellbox.get_bounds(),
                                           closed=True, fill=True, color='white', alpha=1))
                            if cellbox.get_value('iceArea') < 0.8:
                                axis.add_patch(
                                    MatplotPolygon(cellbox.get_bounds(),
                                                   closed=True, fill=True, color='grey',
                                                   alpha=(1 - cellbox.get_value('iceArea'))))
                    else:
                        axis.add_patch(MatplotPolygon(cellbox.get_bounds(), closed=True,
                                       fill=True, color='white', alpha=cellbox.get_value('iceArea')))

                # plot land
                if self._j_grid:
                    if cellbox.land_locked:
                        axis.add_patch(MatplotPolygon(cellbox.get_bounds(),
                                                      closed=True, fill=True, facecolor='lime'))
                else:
                    if cellbox.contains_land():
                        axis.add_patch(MatplotPolygon(cellbox.get_bounds(),
                                                      closed=True, fill=True, facecolor='mediumseagreen'))

                # plot currents
                if plot_currents:
                    axis.quiver((cellbox.long + cellbox.width / 2),
                                (cellbox.lat + cellbox.height / 2),
                                cellbox.get_value('uC'), cellbox.get_value('vC'),
                                scale=1, width=0.002, color='gray')

                # plot borders
                if plot_borders:
                    axis.add_patch(MatplotPolygon(cellbox.get_bounds(),
                                                  closed=True, fill=False, edgecolor='black'))

        # plot highlighted cells
        for colour in highlight_cellboxes:
            for cellbox in highlight_cellboxes[colour]:
                axis.add_patch(MatplotPolygon(cellbox.get_bounds(),
                                              closed=True,
                                              fill=False,
                                              edgecolor=colour,
                                              linewidth=0.5 + len(list(highlight_cellboxes.keys())) -
                                              list(highlight_cellboxes.keys()).index(colour)))

        # plot paths if supplied
        if paths is not None:
            for path in paths:
                if path['Time'] == np.inf:
                    continue
                points = np.array(path['Path']['Points'])
                if routepoints:
                    axis.plot(points[:, 0], points[:, 1], linewidth=3.0, color='b')
                    axis.scatter(points[:, 0], points[:, 1], 30, zorder=99, color='b')
                else:
                    axis.plot(points[:, 0], points[:, 1], linewidth=3.0, color='b')

        if waypoints is not None:
            axis.scatter(waypoints['Long'], waypoints['Lat'], 150, marker='^', color='r')

        axis.set_xlim(self._long_min, self._long_max)
        axis.set_ylim(self._lat_min, self._lat_max)

    def get_neighbour_case(self, cellbox_a, cellbox_b):
        """
            Given two cellBoxes (cellbox_a, cellbox_b) returns a case number
            representing where the two cellBoxes are touching.

            Args:
                cellbox_a (CellBox): starting CellBox
                cellbox_b (CellBox): destination CellBox

            Returns:
                case (int): an int representing the direction of the adjacency
                    between input cellbox_a and cellbox_b. The meaning of each case
                    is as follows -

                        case 0 -> CellBoxes are not neighbours

                        case 1 -> cellbox_b is the North-East corner of cellbox_a\n
                        case 2 -> cellbox_b is East of cellbox_a\n
                        case 3 -> cellbox_b is the South-East corner of cellbox_a\n
                        case 4 -> cellbox_b is South of cellbox_a\n
                        case -1 -> cellbox_b is the South-West corner of cellbox_a\n
                        case -2 -> cellbox_b is West of cellbox_a\n
                        case -3 -> cellbox_b is the North-West corner of cellbox_a\n
                        case -4 -> cellbox_b is North of cellbox_a\n
        """

        if (cellbox_a.long + cellbox_a.width) == cellbox_b.long and (
                cellbox_a.lat + cellbox_a.height) == cellbox_b.lat:
            return 1  # North-East
        if (cellbox_a.long + cellbox_a.width == cellbox_b.long) and (
                cellbox_b.lat < (cellbox_a.lat + cellbox_a.height)) and (
                (cellbox_b.lat + cellbox_b.height) > cellbox_a.lat):
            return 2  # East
        if (cellbox_a.long + cellbox_a.width) == cellbox_b.long and (
                cellbox_a.lat == cellbox_b.lat + cellbox_b.height):
            return 3  # South-East
        if ((cellbox_b.lat + cellbox_b.height) == cellbox_a.lat) and (
                (cellbox_b.long + cellbox_b.width) > cellbox_a.long) and (
                cellbox_b.long < (cellbox_a.long + cellbox_a.width)):
            return 4  # South
        if cellbox_a.long == (cellbox_b.long + cellbox_b.width) and cellbox_a.lat == (
                cellbox_b.lat + cellbox_b.height):
            return -1  # South-West
        if (cellbox_b.long + cellbox_b.width == cellbox_a.long) and (
                cellbox_b.lat < (cellbox_a.lat + cellbox_a.height)) and (
                (cellbox_b.lat + cellbox_b.height) > cellbox_a.lat):
            return -2  # West
        if cellbox_a.long == (cellbox_b.long + cellbox_b.width) and (
                cellbox_a.lat + cellbox_a.height == cellbox_b.lat):
            return -3  # North-West
        if (cellbox_b.lat == (cellbox_a.lat + cellbox_a.height)) and (
                (cellbox_b.long + cellbox_b.width) > cellbox_a.long) and (
                cellbox_b.long < (cellbox_a.long + cellbox_a.width)):
            return -4  # North
        return 0  # Cells are not neighbours.

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

    # Functions used for j_grid regression testing
    def dump_mesh(self, file_location):
        """
            creates a string representation of this Mesh which
            is then saved to a file location specified by parameter
            'file_location'

            for use in j_grid regression testing
        """
        mesh_dump = ""
        for cell_box in self.cellboxes:
            if isinstance(cell_box, CellBox):
                mesh_dump += cell_box.mesh_dump()

        file = open(file_location, "w")
        file.write(mesh_dump)
        file.close()

    def dump_graph(self, file_location):
        """
            creates a string representation of the neighbour relation
            of this Mesh which is then saved to a file location#
            specified by parameter 'file_location'

            for use in j_grid regression testing
        """
        graph_dump = ""

        max_ice_area = 0.8

        for cellbox in self.cellboxes:
            if isinstance(cellbox, CellBox):
                if (not cellbox.land_locked) and cellbox.get_value('SIC') < max_ice_area:
                    graph_dump += cellbox.node_string()

                    cellbox_indx = self.cellboxes.index(cellbox)

                    # case -3 neighbours
                    nw_neighbour_indx = self.neighbour_graph[cellbox_indx][-3]
                    for neighbour in nw_neighbour_indx:
                        if (not self.cellboxes[neighbour].land_locked) and (
                                self.cellboxes[neighbour].get_value('SIC') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].node_string() + ":-3"
                    # case -2 neighbours
                    w_neighbours_indx = self.neighbour_graph[cellbox_indx][-2]
                    for neighbour in w_neighbours_indx:
                        if (not self.cellboxes[neighbour].land_locked) and (
                                self.cellboxes[neighbour].get_value('SIC') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].node_string() + ":-2"
                    # case -1 neighbours
                    sw_neighbours_indx = self.neighbour_graph[cellbox_indx][-1]
                    for neighbour in sw_neighbours_indx:
                        if (not self.cellboxes[neighbour].land_locked) and (
                                self.cellboxes[neighbour].get_value('SIC') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].node_string() + ":-1"
                    # case -4 neighbours
                    n_neighbour_indx = self.neighbour_graph[cellbox_indx][-4]
                    for neighbour in n_neighbour_indx:
                        if (not self.cellboxes[neighbour].land_locked) and (
                            self.cellboxes[neighbour].get_value('SIC') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].node_string() + ":-4"
                    # case 4 neighbours
                    s_neighbours_indx = self.neighbour_graph[cellbox_indx][4]
                    for neighbour in s_neighbours_indx:
                        if (not self.cellboxes[neighbour].land_locked) and (
                                self.cellboxes[neighbour].get_value('SIC') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].node_string() + ":4"
                    # case 1 neighbours
                    ne_neighbour_indx = self.neighbour_graph[cellbox_indx][1]
                    for neighbour in ne_neighbour_indx:
                        if (not self.cellboxes[neighbour].land_locked) and (
                                self.cellboxes[neighbour].get_value('SIC') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].node_string() + ":1"
                    # case 2 neighbours
                    e_neighbour_indx = self.neighbour_graph[cellbox_indx][2]
                    for neighbour in e_neighbour_indx:
                        if (not self.cellboxes[neighbour].land_locked) and (
                                self.cellboxes[neighbour].get_value('SIC') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].node_string() + ":2"
                    # case 3 neighbours
                    se_neighbour_indx = self.neighbour_graph[cellbox_indx][3]
                    for neighbour in se_neighbour_indx:
                        if (not self.cellboxes[neighbour].land_locked) and (
                                self.cellboxes[neighbour].get_value('SIC') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].node_string() + ":3"

                    graph_dump += "\n"

        file = open(file_location, "w")
        file.write(graph_dump)
        file.close()

    def add_current_points(self, current_points):
        """
            Takes a dataframe containing current points and assigns
            then to cellboxes within the Mesh
        """
        for cellbox in self.cellboxes:
            long_loc = current_points.loc[(current_points['long'] > cellbox.long) & (
                current_points['long'] <= (cellbox.long + cellbox.width))]
            lat_long_loc = long_loc.loc[(long_loc['lat'] > cellbox.lat) & (
                long_loc['lat'] <= (cellbox.lat + cellbox.height))]

            cellbox.add_current_points(lat_long_loc)

            # find data point closest to centre to determin land
            def closest_point(df, lat, long):
                n_lat_df = df.iloc[(df['lat'] - lat).abs().argsort()[:15]]
                n_long_df = n_lat_df.iloc[(n_lat_df['long'] - long).abs().argsort()[:1]]
                return n_long_df

            point = closest_point(lat_long_loc, cellbox.getcy(), cellbox.getcx())
            if np.isnan(point['uC'].mean()) or np.isnan(point['vC'].mean()):
                cellbox.land_locked = True

            #cellbox.set_land()

    def cellbox_by_node_string(self, node_string):
        """
            given a node string specified by parameter 'node_string'
            returns a cellbox object which that node string identifies

            used for debugging of j_grids.
        """
        for cellbox in self.cellboxes:
            if isinstance(cellbox, CellBox):
                if cellbox.node_string() == node_string:
                    return cellbox

    @property
    def config(self):
        return self._config