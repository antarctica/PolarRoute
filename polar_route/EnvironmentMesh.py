
import json
from polar_route.AggregatedJGridCellBox import AggregatedJGridCellBox
class EnvironmentMesh:
    """
    a class that defines the environmental mesh structure and contains each cellbox aggregate information


    Attributes:
        bounds (Boundary): the boundaries of this mesh 
        agg_cellboxes (AggregatedCellBox[]): a list of aggregated cellboxes
        neighbour_graph(NeighbourGraph): an object contains each cellbox neighbours information 
        config (dict): conatins the initial config used to build this mesh
        

    
    """
    @classmethod
    def from_json(self, config_file):

        """
            Constructs an Env.Mesh from a given config file to be used by other modules (ex.Vessel Performance Modeller).

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
                                    ], \n
                                "splitting":
                                {
                                     "split_depth": ()real,
                                     "minimum_datapoints": (real)
                                }\n
                                "cellboxes": [
                                    {
                                        
                                    },\n
                                    ...,\n
                                    {...}

                                ]\n,
                                "neighbour_graph": [
                                    {
                                        
                                    },\n
                                    ...,\n
                                    {...}\n
                                ]
                            }\n
                        }\n
                    }\n

           
        """
        with open (file_path , "r") as config_file:
            json_file = json.load(config_file)
        self.config = json_file['config']
        cellboxes_json = json_file['cellboxes']
        self.agg_cellboxes = []
        self.bounds= Boundary (self.config)
        #load the agg_cellboxes
        for cellbox_json in cellboxes_json:
            agg_cellbox = AggregatedCellBox.from_json(cellbox_json)
            self.agg_cellboxes.append (agg_cellbox)
        self.neighbour_graph = NeighbourGraph.from_json (json_file['neighbour_graph'])
        return self
        
    


    def __init__(self, bounds, agg_cellboxes , neighbour_graph ,config):
        """

            Args:
              bounds (Boundary): the boundaries of this mesh 
              agg_cellboxes (AggregatedCellBox[]): a list of aggregated cellboxes
              neighbour_graph(NeighbourGraph): an object contains each cellbox neighbours information 
              config (dict): conatins the initial config used to build this mesh
                
        """
       
        self.bounds = bounds
        self.agg_cellboxes = agg_cellboxes
        self.neighbour_graph = neighbour_graph
        self.config = config
        

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
        output["cellboxes"] = self.cellboxes_to_json()
        output['neighbour_graph'] = self.neighbour_graph.get_graph()

        return json.loads(json.dumps(output))

    def cellboxes_to_json (self):
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
        
        cellboxes_json = []
        for cellbox in self.agg_cellboxes:

                # Get json for CellBox
                cell = cellbox.to_json()
                # Append ID to CellBox
                #cell['id'] = str(self.cellboxes.index(cellbox))

                cellboxes_json.append(cell)
        return cellboxes_json





    def save (self, path):

       with open(path, 'w') as f:
            if  isinstance (self.agg_cellboxes[0] , AggregatedJGridCellBox) :
                self.dump_mesh (f)
            else:
                json.dump(self.to_json(), f)


    def dump_mesh(self, file):
        """
            creates a string representation of this Mesh which
            is then saved to a file location specified by parameter
            'file'
            for use in j_grid regression testing
        """
        mesh_dump = ""
        for cell_box in self.agg_cellboxes:
            if isinstance(cell_box, AggregatedJGridCellBox):
                mesh_dump += cell_box.mesh_dump()

        file.write(mesh_dump)
        file.close()
