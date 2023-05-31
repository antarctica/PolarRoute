

from polar_route.mesh_generation.direction import Direction


class NeighbourGraph:
    """
    A NeighbourGraph is a class that defines the  graphical representation of the adjacency relationship between CellBoxes in the Mesh.\n


    Attributes:
        neighbour_graph (dict): a dictionary that contains cellboxes ids along with their adjacent neighbours in the following form \n

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

    def __init__(self, cellboxes=None, grid_width=0):
        # initialize graph with an empty one
        self.neighbour_graph = {}
        if cellboxes is None:
            cellboxes = []
        self.initialise_neighbour_graph(cellboxes, grid_width)
        self._is_global_mesh = False

    @classmethod
    def from_json(cls, ng_json):
        ''' 
            method that initializes a graph from a json object
            Args: 
            ng_json (json_object): json object that contains the neighbour_graph data
        '''
        neighbour_graph = {}
        for key in ng_json:
            neighbour_graph[key] = ng_json[key]
        obj = NeighbourGraph()
        obj.neighbour_graph = neighbour_graph
        return obj

    def get_graph(self):
        """
        returns the graph dict
        """
        return self.neighbour_graph

    def update_neighbour(self, index, direction, neighbours):
        """
        updates the neighbour in a certain direction
        """
        self.neighbour_graph[index][direction] = neighbours

    def add_neighbour(self, index, direction, neighbour_indx):
        """
        adds a neighbour in a certain direction

        Args:
        index (int): the index of the cellbox to be updated
        direction (int): the direction into which the neighbour will be added
        neighbour_indx (int): the index of the cellbox to be added as a neighbour
        """
        self.neighbour_graph[index][direction].append (neighbour_indx)

    def remove_node_and_update_neighbours(self, cellbox_index):
        ''' 
            method that removes a node in the neighbour_graph at a given index. remove_node_from_neighbours should be called first.
            Args: 
            cellbox_index (int): the index of the cellbox that will get removed from the neighbour_graph
        '''
        # go through all the neighbours in all the directions to remove the give cellbox_index from their neighbour_map
        direction_obj = Direction()
        for direction in direction_obj.__dict__.values():
            self.remove_node_from_neighbours(cellbox_index, direction)

        self.neighbour_graph.pop(cellbox_index)

    def get_neighbours(self, cellbox_indx, direction):
        """
        returns neighbour in a certain direction
        """
        return self.neighbour_graph[cellbox_indx][direction]

    def add_node(self, index, neighbour_map):
        '''
            method that adds a node to the neighbour_graph at a given index
            Args:
            index(int):the index at which the node will be added in the neighbour_graph
            neighbour_map (dict): a dict that contains the neighbours of the node
        '''
        self.neighbour_graph[index] = neighbour_map

    def remove_node(self, cellbox_index):
        '''
            method that removes a node to the neighbour_graph at a given index
            Args: 
            cellbox_index (int): the index of the cellbox that will get removed from the neoghbpur_graph
        '''
        self.neighbour_graph.pop(cellbox_index)

    def update_neighbours(self, cellbox_indx, new_neighbours_indx, direction, cellboxes):
        '''
            method that updates the neighbour of a certain cellbox in a specific direction. It removes cellbox_indx from the neighbour_map of its neighbours in a specific direction and add new_neighbour_indx

            Args: 
                cellbox_index (int): index of the cellbox that its neighbour will be updated
                new_neighbour_indx (int): the index of the new neighbour that will replace cellbox_index
                direction (int): an int that represents the direction of the neighbours that will get updated (e.g. north, south ,..)
                cellboxes(list<CellBox>): the list that contains all the cellboxes of the mesh

        '''
        self.remove_node_from_neighbours(cellbox_indx, direction)
        neighbour_indx_list = self.neighbour_graph[cellbox_indx][direction]
        for indx in neighbour_indx_list:
            crossing_case = self.get_neighbour_case(cellboxes[indx],
                                                    cellboxes[new_neighbours_indx[0]])

            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(
                    new_neighbours_indx[0])

            crossing_case = self.get_neighbour_case(cellboxes[indx],
                                                    cellboxes[new_neighbours_indx[1]])

            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(
                    new_neighbours_indx[1])

    def remove_node_from_neighbours(self, cellbox_indx, direction):
        '''
            method that goes through neighbours in a given direction and remove cellbox_index from their neighbour_maps
            Args:
            cellbox_indx (int): the index of the cellbox that we will go through its neighbours and remove this index from their neighbour_map
            direction (int): an int that represents the direction of the neighbours that will get updated (e.g. north, south ,..)
        '''

        # Try with int keys first as in mesh construction then try with str keys for json mesh
        try:
            neighbour_indx_list = self.neighbour_graph[cellbox_indx][direction]
        except KeyError:
            neighbour_indx_list = self.neighbour_graph[cellbox_indx][str(direction)]

        for indx in neighbour_indx_list:
            try:
                self.neighbour_graph[indx][-1*direction].remove(cellbox_indx)
            except KeyError:
                self.neighbour_graph[str(indx)][str(-1*direction)].remove(int(cellbox_indx))

    def update_corner_neighbours(self, cellbox_indx, north_west_indx, north_east_indx, south_west_indx, south_east_indx):
        '''
            method that updates the corner neighbours of cellbox_indx with the given indeces
        '''
        north_east_corner_indx = self.neighbour_graph[cellbox_indx][Direction.north_east]
        if len(north_east_corner_indx) > 0:
            self.neighbour_graph[north_east_corner_indx[0]][Direction.south_west] = [north_east_indx]

        north_west_corner_indx = self.neighbour_graph[cellbox_indx][Direction.north_west]
        if len(north_west_corner_indx) > 0:
            self.neighbour_graph[north_west_corner_indx[0]][Direction.south_east] = [north_west_indx]

        south_east_corner_indx = self.neighbour_graph[cellbox_indx][Direction.south_east]
        if len(south_east_corner_indx) > 0:
            self.neighbour_graph[south_east_corner_indx[0]][Direction.north_west] = [south_east_indx]

        south_west_corner_indx = self.neighbour_graph[cellbox_indx][Direction.south_west]
        if len(south_west_corner_indx) > 0:
            self.neighbour_graph[south_west_corner_indx[0]][Direction.north_east] = [south_west_indx]
    
    def get_neighbour_case(self, cellbox_a, cellbox_b):
        """
            Given two cellboxes (cellbox_a, cellbox_b) returns a case number
            representing where the two cellboxes are touching.

            Args:
                cellbox_a (CellBox): starting CellBox
                cellbox_b (CellBox): destination CellBox

            Returns:
                int: an int representing the direction of the adjacency between input cellbox_a and cellbox_b. The meaning of each case is as follows -

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
   
        long_a = cellbox_a.bounds.get_long_min()
        lat_a = cellbox_a.bounds.get_lat_min()
        long_b = cellbox_b.bounds.get_long_min()
        lat_b = cellbox_b.bounds.get_lat_min()
        def on_global_bound(cellbox_a , cellbox_b):
            """
            Given two cellboxes (cellbox_a, cellbox_b) returns a boolean
            representing whether the two cellboxes are touching on the global bound (-180,180).

            Args:
                cellbox_a (CellBox): starting CellBox
                cellbox_b (CellBox): destination CellBox

            Returns:
                bool: a boolean representing if the two cellboxes are touching on the global bound (-180,180).            
            """
            return long_a == -180  and cellbox_b.bounds.get_long_max() == 180 or long_b == -180 and cellbox_a.bounds.get_long_max() == 180 
        if self.is_global_mesh() and on_global_bound (cellbox_a , cellbox_b) :
            return self.get_global_mesh_neighbour_case(cellbox_a, cellbox_b)
        if (long_a + cellbox_a.bounds.get_width()) == long_b and (
                lat_a + cellbox_a.bounds.get_height()) == lat_b:
            return Direction.north_east
        if (long_a + cellbox_a.bounds.get_width() == long_b) and (
                lat_b < (lat_a + cellbox_a.bounds.get_height())) and (
                (lat_b + cellbox_b.bounds.get_height()) > lat_a):
            return Direction.east
        if (long_a + cellbox_a.bounds.get_width()) == long_b and (
                lat_a == lat_b + cellbox_b.bounds.get_height()):
            return Direction.south_east
        if ((lat_b + cellbox_b.bounds.get_height()) == lat_a) and (
                (long_b + cellbox_b.bounds.get_width()) > long_a) and (
                long_b < (long_a + cellbox_a.bounds.get_width())):
            return Direction.south
        if long_a == (long_b + cellbox_b.bounds.get_width()) and lat_a == (
                lat_b + cellbox_b.bounds.get_height()):
            return Direction.south_west
        if (long_b + cellbox_b.bounds.get_width() == long_a) and (
                lat_b < (lat_a + cellbox_a.bounds.get_height())) and (
                (lat_b + cellbox_b.bounds.get_height()) > lat_a):
            return Direction.west
        if long_a == (long_b + cellbox_b.bounds.get_width()) and (
                lat_a + cellbox_a.bounds.get_height() == lat_b):
            return Direction.north_west
        if (lat_b == (lat_a + cellbox_a.bounds.get_height())) and (
                (long_b + cellbox_b.bounds.get_width()) > long_a) and (
                long_b < (long_a + cellbox_a.bounds.get_width())):
            return Direction.north
        return 0  # Cells are not neighbours.
    
    def get_global_mesh_neighbour_case(self, cellbox_a, cellbox_b):
        """
            Given two cellboxes in a *global mesh* (cellbox_a, cellbox_b) returns a case number
            representing where the two cellboxes are touching.

            Args:
                cellbox_a (CellBox): starting CellBox
                cellbox_b (CellBox): destination CellBox

            Returns:
                int: an int representing the direction of the adjacency between input cellbox_a and cellbox_b. The meaning of each case is as follows -

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
        long_a = cellbox_a.bounds.get_long_min()
        lat_a = cellbox_a.bounds.get_lat_min()
        long_b = cellbox_b.bounds.get_long_min()
        lat_b = cellbox_b.bounds.get_lat_min()
        if (long_a + cellbox_a.bounds.get_width()) == abs(long_b)and (
                lat_a + cellbox_a.bounds.get_height()) == lat_b:
            return Direction.north_east
        if (long_a + cellbox_a.bounds.get_width() == abs(long_b)) and (
                lat_b < (lat_a + cellbox_a.bounds.get_height())) and (
                (lat_b + cellbox_b.bounds.get_height()) > lat_a):
            return Direction.east
        if (long_a + cellbox_a.bounds.get_width()) == abs(long_b) and (
                lat_a == lat_b + cellbox_b.bounds.get_height()):
            return Direction.south_east
        if ((lat_b + cellbox_b.bounds.get_height()) == lat_a) and (
                (long_b + cellbox_b.bounds.get_width()) > long_a) and (
                long_b < (long_a + cellbox_a.bounds.get_width())):
            return Direction.south
        if abs(long_a) == (long_b + cellbox_b.bounds.get_width()) and lat_a == (
                lat_b + cellbox_b.bounds.get_height()):
            return Direction.south_west
        if (long_b + cellbox_b.bounds.get_width() == abs(long_a)) and (
                lat_b < (lat_a + cellbox_a.bounds.get_height())) and (
                (lat_b + cellbox_b.bounds.get_height()) > lat_a):
            return Direction.west
        if abs(long_a) == (long_b + cellbox_b.bounds.get_width()) and (
                lat_a + cellbox_a.bounds.get_height() == lat_b):
            return Direction.north_west
        if (lat_b == (lat_a + cellbox_a.bounds.get_height())) and (
                (long_b + cellbox_b.bounds.get_width()) > long_a) and (
                long_b < (long_a + cellbox_a.bounds.get_width())):
            return Direction.north
        return 0  # Cells are not neighbours.

    def remove_neighbour(self,  index,  direction, neighbour_index):
        """ 
            remove certain neighbour in a specific direction
        """
        self.neighbour_graph[index][direction].remove(neighbour_index)

    def initialise_neighbour_graph(self, cellboxes, grid_width):
        """
         initialize the neighbour graph
        """
        for cellbox in cellboxes:
            cellbox_indx = cellboxes.index(cellbox)
            neighbour_map = self.initialise_map(
                cellbox_indx, grid_width, len(cellboxes))

            self.add_node(cellbox_indx, neighbour_map)

    def initialise_map(self, cellbox_indx, grid_width, cellboxes_length):
        """
            initialse the neighbour map of a cellbox with the given cellbox_index
        """
        neighbour_map = {Direction.north_east: [],
                         Direction.east: [],
                         Direction.south_east: [],
                         Direction.south: [],
                         Direction.south_west: [],
                         Direction.west: [],
                         Direction.north_west: [], 
                         Direction.north: []}

        # add east neighbours to neighbour graph
        if (cellbox_indx + 1) % grid_width != 0:
            neighbour_map[Direction.east].append(cellbox_indx + 1)
            # south-east neighbours
            if cellbox_indx + grid_width < cellboxes_length:
                neighbour_map[Direction.north_east].append(
                    int((cellbox_indx + grid_width) + 1))
            # north-east neighbours
            if cellbox_indx - grid_width >= 0:
                neighbour_map[Direction.south_east].append(
                    int((cellbox_indx - grid_width) + 1))

        # add west neighbours to neighbour graph
        if cellbox_indx % grid_width != 0:
            neighbour_map[Direction.west].append(cellbox_indx - 1)
            # add south-west neighbours to neighbour graph
            if cellbox_indx + grid_width < cellboxes_length:
                neighbour_map[Direction.north_west].append(
                    int((cellbox_indx + grid_width) - 1))
            # add north-west neighbours to neighbour graph
            if cellbox_indx - grid_width >= 0:
                neighbour_map[Direction.south_west].append(
                    int((cellbox_indx - grid_width) - 1))

        # add south neighbours to neighbour graph
        if cellbox_indx + grid_width < cellboxes_length:
            neighbour_map[Direction.north].append(int(cellbox_indx + grid_width))

        # add north neighbours to neighbour graph
        if cellbox_indx - grid_width >= 0:
            neighbour_map[Direction.south].append(
                int(cellbox_indx - grid_width))
        return neighbour_map
    
    def set_global_mesh (self, is_global):
        self._is_global_mesh = is_global
    def is_global_mesh (self):
        return self._is_global_mesh