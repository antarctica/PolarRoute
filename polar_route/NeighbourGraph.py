




class NeighbourGraph:
    """
    A NeighbourGraph is a class that defines the  graphical representation of the adjacency \n relationship between CellBoxes in the Mesh.


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
   

    def __init__(self , cellboxes ,grid_width):
        # initialize graph with an empty one
        self.neighbour_graph = {}
        self.initialise_neighbour_graph(cellboxes ,grid_width)


    def get_graph(self):
        
        return self.neighbour_graph


    def update_neighbour(self ,  index ,  direction , neighbours):
        self.neighbour_graph [index][direction] = neighbours     
        
    def get_neighbours(self ,cellbox_indx , direction):
        return self.neighbour_graph [cellbox_indx][direction] 

    def add_node(self ,  index ,  neighbour_map):
        self.neighbour_graph [index] = neighbour_map     
       
    
        
    def remove_node (self, cellbox_index):
      self.neighbour_graph.pop(cellbox_index)

 
    def update_neighbours(self,cellbox_indx, new_neighbours_indx, direction, cellboxes):
        '''
        method that updates the neighbour of a certain cellbox in a specific direction
        '''
          
        self.remove_node_from_neighbours (cellbox_indx, direction)  
        neighbour_indx_list = self.neighbour_graph[cellbox_indx][direction]
        for indx in neighbour_indx_list:
            crossing_case = self.get_neighbour_case(cellboxes[indx],
                                                    cellboxes[new_neighbours_indx[0]])
            
            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(new_neighbours_indx[0])

            crossing_case = self.get_neighbour_case(cellboxes[indx],
                                                    cellboxes[new_neighbours_indx[1]])
           
            if crossing_case != 0:
                self.neighbour_graph[indx][crossing_case].append(new_neighbours_indx[1])   

    def remove_node_from_neighbours (self , cellbox_indx, direction): # go through neighbours in a given direction and remove cellbox_index from their neighbour_maps
        neighbour_indx_list = self.neighbour_graph[cellbox_indx][direction]
        for indx in neighbour_indx_list:
            self.neighbour_graph[indx][-1*direction].remove(cellbox_indx)


    def update_corner_neighbours(self, cellbox_indx, north_west_indx, north_east_indx, south_west_indx, south_east_indx):
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

        long_a = cellbox_a.bounds.get_long_min()
        lat_a = cellbox_a.bounds.get_lat_min()
        long_b = cellbox_b.bounds.get_long_min()
        lat_b = cellbox_b.bounds.get_lat_min()
        if (long_a + cellbox_a.bounds.get_width()) == long_b and (
                lat_a + cellbox_a.bounds.get_height()) == lat_b:
            return 1  # North-East
        if (long_a + cellbox_a.bounds.get_width ()== long_b) and (
                lat_b < (lat_a + cellbox_a.bounds.get_height())) and (
                (lat_b + cellbox_b.bounds.get_height()) > lat_a):
            return 2  # East
        if (long_a + cellbox_a.bounds.get_width()) == long_b and (
                lat_a == lat_b + cellbox_b.bounds.get_height()):
            return 3  # South-East
        if ((lat_b + cellbox_b.bounds.get_height()) == lat_a) and (
                (long_b + cellbox_b.bounds.get_width()) > long_a) and (
                long_b < (long_a + cellbox_a.bounds.get_width())):
            return 4  # South
        if long_a == (long_b + cellbox_b.bounds.get_width()) and lat_a == (
                lat_b + cellbox_b.bounds.get_height()):
            return -1  # South-West
        if (long_b + cellbox_b.bounds.get_width() == long_a) and (
                lat_b < (lat_a + cellbox_a.bounds.get_height())) and (
                (lat_b + cellbox_b.bounds.get_height()) > lat_a):
            return -2  # West
        if long_a == (long_b + cellbox_b.bounds.get_width()) and (
                lat_a + cellbox_a.bounds.get_height() == lat_b):
            return -3  # North-West
        if (lat_b == (lat_a + cellbox_a.bounds.get_height())) and (
                (long_b + cellbox_b.bounds.get_width()) > long_a) and (
                long_b < (long_a + cellbox_a.bounds.get_width())):
            return -4  # North
        return 0  # Cells are not neighbours.




    def remove_neighbour(self ,  index ,  direction , neighbour_index):
        self.neighbour_graph [index][direction].remove (neighbour_index) 

    def initialise_neighbour_graph (self , cellboxes ,grid_width):
        
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

            self.add_node (cellbox_indx , neighbour_map)
    