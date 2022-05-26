from tracemalloc import start
import numpy as np
from RoutePlanner.CellBox import CellBox
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MatplotPolygon
import math
import xarray as xr
from shapely.geometry import Polygon
import geopandas as gpd

class CellGrid:
    """
        TODO
    """

    def __init__(self, config, j_grid=False):
        self.config = config

        self._long_min = config['Region']['longMin']
        self._long_max = config['Region']['longMax']
        self._lat_min = config['Region']['latMin']
        self._lat_max = config['Region']['latMax']

        self._cell_width = config['Region']['cellWidth']
        self._cell_height = config['Region']['cellHeight']

        self._start_time = config['Region']['startTime']
        self._end_time = config['Region']['endTime']

        self._data_sources = config['Data_sources']

        self._j_grid = j_grid

        self.cellboxes = []

        # Initialise cellBoxes.
        for lat in np.arange(self._lat_min, self._lat_max, self._cell_height):
            for long in np.arange(self._long_min, self._long_max, self._cell_width):
                cellbox = CellBox(lat, long, self._cell_width, self._cell_height,
                                    splitting_conditions = [], j_grid = self._j_grid)
                self.cellboxes.append(cellbox)

        grid_width = (self._long_max - self._long_min) / self._cell_width
        grid_height = (self._lat_max - self._lat_min) / self._cell_height

        # Calculate initial neighbours graph.
        self.neighbour_graph = {}
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
            if (cellbox_indx) % grid_width != 0:
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

            # set gridCoord of cellBox
            x_coord = cellbox_indx % grid_width
            y_coord = abs(math.floor(cellbox_indx / grid_width) - (grid_height - 1))
            cellbox.set_grid_coord(x_coord, y_coord)

            # set focus of cellBox
            cellbox.set_focus([])
        self.splitting_conditions = []
        for data_source in config['Data_sources']:
            self.add_data_source(data_source)

        self.iterative_split(config['Region']["splitDepth"])

    # Functions for adding data to the cellgrid
    def add_data_source(self, data_source):
        """
            injest data from a given data source into the cellgrid structure.

            data_source is given in the format:
            {
                "path": "xxx.nc"
                "latName":"x",
                "longName":"x",
                "timeName":"x",
                "values":[
                    {
                        "sourceName":"x",
                        "destinationName":"x",
                        (OPTIONAL)"splittingCondition":{
                            "threshold":x,
                            "lowerBound"x,
                            "upperBound"x
                        }
                    }
                ]
            }

        """
        for cellbox in self.cellboxes:
            lat_min = cellbox.lat
            long_min = cellbox.long

            long_max = long_min + cellbox.width
            lat_max = lat_min + cellbox.height

            path = data_source['path']

            dataset = xr.open_dataset(path)

            if "timeName" in data_source:
                dataset = dataset.rename({data_source['latName']:'lat',
                                        data_source['longName']:'long',
                                        data_source['timeName']:'time'})

                data_slice = dataset.sel(time = slice(self._start_time, self._end_time),
                                        lat = slice(lat_min, lat_max),
                                        long = slice(long_min, long_max))
            else:
                dataset = dataset.rename({data_source['latName']:'lat',
                                        data_source['longName']:'long'})

                data_slice = dataset.sel(lat = slice(lat_min, lat_max),
                                        long = slice(long_min, long_max))

            dataframe = data_slice.to_dataframe()
            dataframe = dataframe.reset_index()

            selected = []
            for value in data_source['values']:
                dataframe=dataframe.rename(columns={value['sourceName']:value['destinationName']})
                selected = selected + [value['destinationName']]

                if "splittingCondition" in value:
                    splitting_condition = {value['destinationName'] : value['splittingCondition']}
                    cellbox.add_splitting_condition(splitting_condition)

            dataframe = dataframe.dropna(subset = selected)

            cellbox.add_data_points(dataframe)

    def add_data_points(self, data_points):
        """
            takes a dataframe containing geospatial-temporal located values and assigns them to
            cellboxes within this cellgrid.
        """
        for cell_box in self.cellboxes:
            long_loc = data_points.loc[(data_points['long'] > cell_box.long) &
                (data_points['long'] <= (cell_box.long + cell_box.width))]
            lat_long_loc = long_loc.loc[(long_loc['lat'] > cell_box.lat) &
                (long_loc['lat'] <= (cell_box.lat + cell_box.height))]

            cell_box.add_data_points(lat_long_loc)

    # Functions for outputting the cellgrid
    def output_dataframe(self):
        """
            requires rework as to not used hard-coded data types.
        """
        cellgrid_dataframe = []

        for idx,cellbox in enumerate(self.cellboxes):
            if isinstance(cellbox, CellBox):
                # # Don't append cell if Ice or above threshold
                # if c.get_value('iceArea') >= self.config['Vehicle_Info']['MaxIceExtent']:
                #     continue
                # if self._j_grid:
                #     if c.isLandM():
                #         continue
                # else:
                #     if c.containsLand():
                #         continue


                # Inspecting neighbour graph and outputting in list
                neigh = self.neighbour_graph[idx]
                cases      = []
                neigh_indx = []
                for case in neigh.keys():
                    indxs = neigh[case]
                    if len(indxs) == 0:
                        continue
                    for indx in indxs:
                        if (self.cellboxes[indx].get_value('iceArea')*100 >
                                self.config['Vehicle_Info']['MaxIceExtent']):
                            continue
                        if self._j_grid:
                            if self.cellboxes[indx].isLandM():
                                continue
                        else:
                            if self.cellboxes[indx].contains_land():
                                continue
                        cases.append(case)
                        neigh_indx.append(indx)

                if self._j_grid:
                    is_land = cellbox.is_land_m()
                else:
                    is_land = cellbox.contains_land()

                index_df = pd.Series({'Index':int(idx),
                        'geometry':Polygon(cellbox.get_bounds()),
                        'cell_info':[cellbox.getcx(),
                            cellbox.getcy(),cellbox.getdcx(),cellbox.getdcy()],
                        'case':cases,
                        'neighbourIndex':neigh_indx,
                        'Land':is_land,
                        'Ice Area':cellbox.get_value('iceArea')*100,
                        'Ice Thickness':cellbox.ice_thickness(self.config['Region']['startTime']),
                        'Ice Density':cellbox.ice_density(self.config['Region']['startTime']),
                        'Depth': cellbox.get_value('depth'),
                        'Vector':[cellbox.get_value('uC'),cellbox.get_value('vC')]
                        })

                cellgrid_dataframe.append(index_df)

        cellgrid_dataframe = pd.concat(cellgrid_dataframe,axis=1).transpose()

        ## Cell Further South than -78.0 set to land.
        cellgrid_dataframe['Land'][np.array([x[1] for x in cellgrid_dataframe['cell_info']])
            < -78.0] = True

        cellgrid_dataframe = gpd.GeoDataFrame(cellgrid_dataframe,
            crs={'init': 'epsg:4326'},geometry='geometry')
        return cellgrid_dataframe

    # Functions for spltting cellboxes within the cellgrid
    def split_and_replace(self, cellbox):
        """
            Replaces a cellBox given by parameter 'cellBox' in this grid with
            4 smaller cellBoxes representing the four corners of the given cellBox.
            A neighbours map is then created for each of the 4 new cellBoxes
            and the neighbours map for all surrounding cell boxes is updated.

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

    def iterative_split(self, split_amount):
        """
            Iterates over all cellboxes in the cellGrid a number of times defined by
            parameter 'splitAmount', splitting and replacing each one if it is
            not homogenous, as dictated by function Cellbox.should_be_split()
        """
        for iter in range(0, split_amount):
            self.split_graph()

    def split_graph(self):
        """
            Iterates once over all cellBoxes in the cellGrid,
            splitting and replacing each one if it is not homogenous.
        """
        for indx in range(0, len(self.cellboxes) - 1):
            cellbox = self.cellboxes[indx]
            if isinstance(cellbox, CellBox):
                if cellbox.should_be_split():
                    self.split_and_replace(cellbox)

    # Functions for debugging
    def plot(self, highlight_cellboxes = {}, plot_ice = True, plot_currents = False,
        plot_borders = True, paths=None, routepoints=False,waypoints=None):
        """
            creates and displays a plot for this cellGrid

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
                    if cellbox.landLocked:
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
                    linewidth = 0.5 + len(list(highlight_cellboxes.keys())) -
                        list(highlight_cellboxes.keys()).index(colour)))

        # plot paths if supplied
        if not paths is None:
            for path in paths:
                if path['Time'] == np.inf:
                    continue
                points = np.array(path['Path']['Points'])
                if routepoints:
                    axis.plot(points[:,0],points[:,1],linewidth=3.0,color='b')
                    axis.scatter(points[:,0],points[:,1],30,zorder=99,color='b')
                else:
                    axis.plot(points[:,0],points[:,1],linewidth=3.0,color='b')


        if not waypoints is None:
            axis.scatter(waypoints['Long'],waypoints['Lat'],150,marker='^',color='r')

        axis.set_xlim(self._long_min, self._long_max)
        axis.set_ylim(self._lat_min, self._lat_max)

    def get_neighbour_case(self, cellbox_a, cellbox_b):
        """
            Given two cellBoxes (cellBoxA, cellBoxB) returns a case number
            representing where the two cellBoxes are touching.

            case 0 -> cellBoxes are not neighbours

            case 1 -> cellBoxB is the North-East corner of cellBoxA
            case 2 -> cellBoxB is East of cellBoxA
            case 3 -> cellBoxB is the South-East corner of cellBoxA
            case 4 -> cellBoxB is South of cellBoxA
            case -1 -> cellBoxB is the South-West corner of cellBoxA
            case -2 -> cellBoxB is West of cellBoxA
            case -3 -> cellBoxB is the North-West corner of cellBoxA
            case -4 -> cellBoxB is North of cellBoxA
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
        """
        selected_cell = []
        for cellbox in self.cellboxes:
            if isinstance(cellbox, CellBox):
                if cellbox.contains_point(lat, long):
                    selected_cell.append(cellbox)
        return selected_cell

    def to_json(self):
        """
            Returns this cellGrid converted to JSON format.
        """
        json = "{ \"cellBoxes\":["
        for cellbox in self.cellboxes:
            json += cellbox.toJSON() + ",\n"

        json = json[:-2] # remove last comma and newline
        json += "]}"
        return json

    # Functions used for j_grid regression testing
    def dump_mesh(self, file_location):
        """
            creates a string representaion of this cellgrid which
            is then saved to a file location specifed by parameter
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
            of this cellgrid which is then saved to a file location#
            specified by parameter 'file_location'

            for use in j_grid regression testing
        """
        graph_dump = ""

        max_ice_area = 0.8

        for cellbox in self.cellboxes:
            if isinstance(cellbox, CellBox):
                if (not cellbox.landLocked) and cellbox.get_value('iceArea') < max_ice_area:
                    graph_dump += cellbox.node_string()

                    cellbox_indx = self.cellboxes.index(cellbox)

                    # case -3 neighbours
                    nw_neighbour_indx = self.neighbour_graph[cellbox_indx][-3]
                    for neighbour in nw_neighbour_indx:
                        if (not self.cellboxes[neighbour].landLocked) and (
                                self.cellboxes[neighbour].get_value('iceArea') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].nodeString() + ":-3"
                    # case -2 neighbours
                    w_neighbours_indx = self.neighbour_graph[cellbox_indx][-2]
                    for neighbour in w_neighbours_indx:
                        if (not self.cellboxes[neighbour].landLocked) and (
                                self.cellboxes[neighbour].get_value('iceArea') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].nodeString() + ":-2"
                    # case -1 neighbours
                    sw_neighbours_indx = self.neighbour_graph[cellbox_indx][-1]
                    for neighbour in sw_neighbours_indx:
                        if (not self.cellboxes[neighbour].landLocked) and (
                                self.cellboxes[neighbour].get_value('iceArea') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].nodeString() + ":-1"
                    # case -4 neighbours
                    n_neighbour_indx = self.neighbour_graph[cellbox_indx][-4]
                    for neighbour in n_neighbour_indx:
                        if (not self.cellboxes[neighbour].landLocked) and (
                            self.cellboxes[neighbour].get_value('iceArea') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].nodeString() + ":-4"
                    # case 4 neighbours
                    s_neighbours_indx = self.neighbour_graph[cellbox_indx][4]
                    for neighbour in s_neighbours_indx:
                        if (not self.cellboxes[neighbour].landLocked) and (
                                self.cellboxes[neighbour].get_value('iceArea') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].nodeString() + ":4"
                    # case 1 neighbours
                    ne_neighbour_indx = self.neighbour_graph[cellbox_indx][1]
                    for neighbour in ne_neighbour_indx:
                        if (not self.cellboxes[neighbour].landLocked) and (
                                self.cellboxes[neighbour].get_value('iceArea') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].nodeString() + ":1"
                    # case 2 neighbours
                    e_neighbour_indx = self.neighbour_graph[cellbox_indx][2]
                    for neighbour in e_neighbour_indx:
                        if (not self.cellboxes[neighbour].landLocked) and (
                                self.cellboxes[neighbour].get_value('iceArea') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].nodeString() + ":2"
                    # case 3 neighbours
                    se_neighbour_indx = self.neighbour_graph[cellbox_indx][3]
                    for neighbour in se_neighbour_indx:
                        if (not self.cellboxes[neighbour].landLocked) and (
                                self.cellboxes[neighbour].get_value('iceArea') < max_ice_area):
                            graph_dump += "," + self.cellboxes[neighbour].nodeString() + ":3"

                    graph_dump += "\n"

        file = open(file_location, "w")
        file.write(graph_dump)
        file.close()

    def add_current_points(self, current_points):
        """
            Takes a dataframe containing current points and assigns
            then to cellboxes within the cellgrid
        """
        for cellbox in self.cellboxes:
            long_loc = current_points.loc[(current_points['long'] > cellbox.long) & (
                current_points['long'] <= (cellbox.long + cellbox.width))]
            lat_long_loc = long_loc.loc[(long_loc['lat'] > cellbox.lat) & (
                long_loc['lat'] <= (cellbox.lat + cellbox.height))]

            cellbox.add_current_points(lat_long_loc)
            cellbox.set_land()

    def cellbox_by_node_string(self, node_string):
        """
            given a node string specifed by parameter 'node_string'
            returns a cellbox object which that node string identifies
        """
        for cellbox in self.cellboxes:
            if isinstance(cellbox, CellBox):
                if cellbox.node_string() == node_string:
                    return cellbox
