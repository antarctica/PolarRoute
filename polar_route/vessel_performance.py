"""
The VesselPerformance class deals with all the vehicle specific features of the meshed environment model. It uses the
specific vessel parameters defined in the config to determine which cells in the mesh are inaccessible to a given
vehicle, either based on elevation or sea ice concentration, and determines vessel performance characteristics, such as
the speed and fuel consumption, based on the content of the cellboxes.

The input to the class is the mesh object produced by the CellGrid class in json format and it returns a modified mesh
in the same format via the VesselPerformance.to_json method.

Example:
    An example of how to use this code can be executed by running the following::

        import json
        from RoutePlanner.CellGrid import CellGrid
        from RoutePlanner.vessel_performance import VesselPerformance

        with open("config.json", 'r') as f:
            cfg = json.load(f)

        mesh = CellGrid(cfg)
        mesh_json = mesh.to_json()

        vp = VesselPerformance(mesh_json)

        vehicle_mesh = vp.to_json()
"""
import logging
import json
import numpy as np
import pandas as pd

from polar_route.utils import timed_call


class VesselPerformance:
    """
        Class for modelling the vessel performance.
        Takes a mesh as input in json format and modifies it to include vessel specifics.

        Attributes:
            mesh (dict): A dictionary containing all the mesh information
            config (dict): The config used to generate the input mesh
            mesh_df (DataFrame): The cellbox information from the mesh stored in a pandas DataFrame
            vessel_params (dict): The vessel specific information contained within the config

    """
    def __init__(self, mesh_json):
        """
            Constructs the VesselPerformance class from a given input mesh in json format which is then modified
            according to the vessel parameters defined in the config.

            Args:
                mesh_json (dict): The input mesh containing the cellboxes and neighbour graph as well as the config used
                to generate the mesh.
        """
        logging.info("Initialising vessel performance...")
        self.mesh = mesh_json
        self.config = self.mesh['config']
        self.mesh_df = pd.DataFrame(self.mesh['cellboxes']).set_index('id')
        self.vessel_params = self.config['Vessel']

        # Check for NaNs in input and zero them if present
        if self.mesh_df.isnull().values.any():
            logging.debug("NaNs present in input mesh, setting all NaN values to zero")
            self.mesh_df = self.mesh_df.fillna(0.)

        # Identifying land and extreme ice cells then removing them from the neighbour graph
        self.land()
        self.extreme_ice()
        self.inaccessible_nodes()

        # Checking if the speed is defined in the input mesh
        if 'speed' not in self.mesh_df:
            logging.debug(f'No speed in mesh, assigning default value of {self.vessel_params["Speed"]} '
                          f'{self.vessel_params["Unit"]} from config')
            self.mesh_df['speed'] = self.vessel_params["Speed"]

        # Modify speed based on ice resistance
        self.speed()

        # Calculate fuel usage based on speed and ice resistance
        self.fuel()

        # Check again for NaNs and zero them then warn if present
        if self.mesh_df.isnull().values.any():
            logging.warning("NaNs present in output mesh, setting all NaN values to zero!")
            self.mesh_df = self.mesh_df.fillna(0.)

        # Updating the mesh indexing and cellboxes
        self.mesh_df['id'] = self.mesh_df.index
        self.mesh['cellboxes'] = self.mesh_df.to_dict('records')

    def to_json(self):
        """
            Method to return the modified mesh in json format.

            Returns:
                j_mesh (dict): a dictionary representation of the modified mesh.
        """
        j_mesh = json.loads(json.dumps(self.mesh))
        return j_mesh

    def land(self):
        """
            Method to determine which cells are land based on configured minimum depth.
        """
        if 'elevation' not in self.mesh_df:
            logging.warning("No elevation data in mesh, no cells will be marked as land!")
            self.mesh_df['land'] = False
        else:
            self.mesh_df['land'] = self.mesh_df['elevation'] > self.vessel_params['MinDepth']
            logging.debug(f"{self.mesh_df['land'].sum()} cells inaccessible due to land")

    def extreme_ice(self):
        """
            Method to determine which cells are inaccessible based on configured max ice concentration.
        """
        if 'SIC' not in self.mesh_df:
            logging.info("No sea ice concentration data in mesh")
            self.mesh_df['ext_ice'] = False
        else:
            self.mesh_df['ext_ice'] = self.mesh_df['SIC'] > self.vessel_params['MaxIceExtent']
            logging.debug(f"{self.mesh_df['ext_ice'].sum()} cells inaccessible due to extreme ice")

    @timed_call
    def inaccessible_nodes(self):
        """
            Method to determine which nodes are inaccessible and remove them from the neighbour graph.
        """
        logging.info("Determining which nodes are inaccessible due to land and ice")
        inaccessible = self.mesh_df[(self.mesh_df['land']) | (self.mesh_df['ext_ice'])]
        inaccessible_idx = list(inaccessible.index)

        self.mesh['neighbour_graph'] = self.remove_nodes(self.mesh['neighbour_graph'], inaccessible_idx)

    def ice_resistance(self, velocity, area, thickness, density):
        """
            Method to find the ice resistance force at a given speed in a given cell.

            Args:
                velocity (float): The speed of the vessel in km/h
                area (float): The average sea ice concentration in the cell as a percentage
                thickness (float): The average ice thickness in the cell in m
                density (float): The average ice density in the cell in kg/m^3

            Returns:
                resistance (float): Resistance force in N
        """
        # Model parameters for different hull types
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

        hull = self.vessel_params['HullType']
        beam = self.vessel_params['Beam']
        kparam, bparam, nparam = hull_params[hull]
        gravity = 9.81  # m/s-2

        speed = velocity * (5. / 18.)  # assume km/h and convert to m/s

        froude = speed / np.sqrt(gravity * area / 100 * thickness)
        resistance = 0.5 * kparam * (froude ** bparam) * density * beam * thickness * (speed ** 2) * (
                    (area / 100) ** nparam)
        return resistance

    def inverse_resistance(self, area, thickness, density):
        """
            Method to find the vessel speed that keeps the ice resistance force below a given threshold in a given cell.

            Args:
                area (float): The average sea ice concentration in the cell as a percentage
                thickness (float): The average ice thickness in the cell in m
                density (float): The average ice density in the cell in kg/m^3

            Returns:
                speed (float): Safe vessel speed in km/h
        """
        # Model parameters for different hull types
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}
        # force_limit (float): Resistance force value that should not be exceeded in N
        force_limit = self.vessel_params['ForceLimit']

        hull = self.vessel_params['HullType']
        beam = self.vessel_params['Beam']
        kparam, bparam, nparam = hull_params[hull]
        gravity = 9.81  # m/s-2

        vexp = 2 * force_limit / (kparam * density * beam * thickness * ((area / 100) ** nparam) * (
                    gravity * thickness * area / 100) ** -(bparam / 2))

        vms = vexp ** (1 / (2.0 + bparam))
        speed = vms * (18. / 5.)  # convert from m/s to km/h

        return speed

    @timed_call
    def speed(self):
        """
            Method to compile the new speeds calculated based on the ice resistance into the mesh.
        """
        logging.info("Calculating new speeds based on resistance models")

        if all(k in self.mesh_df for k in ("SIC", "thickness", "density")):
            logging.info("Adjusting speed according to ice resistance model")
            self.mesh_df['ice resistance'] = np.nan
            for idx, row in self.mesh_df.iterrows():
                if row['SIC'] == 0.0:
                    self.mesh_df.loc[idx, 'speed'] = self.vessel_params['Speed']
                    self.mesh_df.loc[idx, 'ice resistance'] = 0.0
                elif row['SIC'] > self.vessel_params['MaxIceExtent']:
                    self.mesh_df.loc[idx, 'speed'] = 0.0
                    self.mesh_df.loc[idx, 'ice resistance'] = np.inf
                else:
                    rp = self.ice_resistance(self.vessel_params['Speed'], row['SIC'], row['thickness'], row['density'])
                    if rp > self.vessel_params['ForceLimit']:
                        new_speed = self.inverse_resistance(row['SIC'], row['thickness'], row['density'])
                        rp = self.ice_resistance(new_speed, row['SIC'], row['thickness'], row['density'])
                        self.mesh_df.loc[idx, 'speed'] = new_speed
                        self.mesh_df.loc[idx, 'ice resistance'] = rp
                    else:
                        self.mesh_df.loc[idx, 'speed'] = self.vessel_params['Speed']
                        self.mesh_df.loc[idx, 'ice resistance'] = rp
        else:
            logging.info("No resistance data available, no speed adjustment necessary")

    def speed_simple(self):
        """
            Method to calculate the speed based on the sea ice concentration using a simple toy model.
        """
        self.mesh_df['speed'] = (1 - np.sqrt(self.mesh_df['SIC'] / 100)) * \
                                        self.vessel_params['Speed']

    @timed_call
    def fuel(self):
        """
            Method to calculate the fuel usage in tons per day based on speed in km/h and ice resistance in N.
        """
        logging.info("Calculating fuel requirements in each cell")

        if 'ice resistance' in self.mesh_df:
            self.mesh_df['fuel'] = (0.00137247 * self.mesh_df['speed'] ** 2 - 0.0029601 *
                                    self.mesh_df['speed'] + 0.25290433
                                    + 7.75218178e-11 * self.mesh_df['ice resistance'] ** 2
                                    + 6.48113363e-06 * self.mesh_df['ice resistance']) * 24.0
        else:
            self.mesh_df['fuel'] = (0.00137247 * self.mesh_df['speed'] ** 2 - 0.0029601 *
                                    self.mesh_df['speed'] + 0.25290433) * 24.0

    def remove_nodes(self, neighbour_graph, inaccessible_nodes):
        """
            Method to remove a list of inaccessible nodes from a given neighbour graph.

            Args:
                neighbour_graph (dict): A dictionary containing indexes of cellboxes and how they are connected

                {
                    'index':{
                        '1':[index,...],
                        '2':[index,...],
                        '3':[index,...],
                        '4':[index,...],
                        '-1':[index,...],
                        '-2':[index,...],
                        '-3':[index,...],
                        '-4':[index,...]
                    },
                    'index':{...},
                    ...
                }

                inaccessible_nodes (list): A list of indexes to be removed from the neighbour_graph

            Returns:
                accessibility_graph (dict): A new neighbour graph with the inaccessible nodes removed
        """
        logging.debug(f"Removing {len(inaccessible_nodes)} nodes from the neighbour graph")
        accessibility_graph = neighbour_graph.copy()

        for node in accessibility_graph.keys():
            for case in accessibility_graph[node].keys():
                for inaccessible_node in inaccessible_nodes:
                    if int(inaccessible_node) in accessibility_graph[node][case]:
                        accessibility_graph[node][case].remove(int(inaccessible_node))

        for node in inaccessible_nodes:
            accessibility_graph.pop(node)

        return accessibility_graph
