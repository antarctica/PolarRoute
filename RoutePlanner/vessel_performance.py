"""
    Class for modelling the vessel performance.
    Takes a mesh as input in json format and modifies it to include vessel specifics.
"""
import json
import numpy as np
import pandas as pd


class VesselPerformance:
    def __init__(self, mesh_json):
        """
            FILL
        """

        self.mesh = json.loads(mesh_json)
        self.config = self.mesh['config']
        self.mesh_df = pd.DataFrame(self.mesh['cellboxes']).set_index('id')
        self.vessel_params = self.config['Vessel']

        # Identifying land and extreme ice cells then removing them from the neighbour graph
        self.land()
        self.extreme_ice()
        self.inaccessible_nodes()

        # Checking if Speed defined in file
        if 'speed' not in self.mesh_df:
            self.mesh_df['speed'] = self.vessel_params["Speed"]

        # Modify speed based on ice resistance
        self.speed()

        # Calculate fuel usage based on speed and ice resistance
        self.fuel()

    def to_json(self):
        """
        Method to return the modified mesh in json format.
        """
        self.mesh['cellboxes'] = self.mesh_df.to_dict('records')
        return json.dumps(self.mesh)

    def land(self):
        self.mesh_df['land'] = self.mesh_df['elevation'] > self.vessel_params['MinDepth']

    def extreme_ice(self):
        self.mesh_df['ext_ice'] = self.mesh_df['SIC'] > self.vessel_params['MaxIceExtent']

    def inaccessible_nodes(self):
        """
        Method to determine which nodes are inaccessible and remove them from the neighbour graph.
        """

        inaccessible = self.mesh_df[(self.mesh_df['land']) | (self.mesh_df['ext_ice'])]
        inaccessible_idx = list(inaccessible.index)

        self.mesh['neighbour_graph'] = self.remove_nodes(self.mesh['neighbour_graph'], inaccessible_idx)

    def ice_resistance(self, velocity, area, thickness, density):
        """
                Function to find the ice resistance force at a given speed in a given cell.

                Inputs:
                cell - Cell box object

                Outputs:
                resistance - Resistance force
        """
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
        Function to find the speed that keeps the ice resistance force below a given threshold.

        Inputs:
        force_limit - Force limit
        cell        - Cell box object

        Outputs:
        speed - Vehicle Speed
        """
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}
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

    def speed(self):
        """
            A function to compile the new speeds calculated based on the ice resistance into the mesh.
        """

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

    def speed_simple(self):
        self.mesh_df['speed'] = (1 - np.sqrt(self.mesh_df['SIC'] / 100)) * \
                                        self.vessel_params['Speed']

    def fuel(self):
        """
        Fuel usage in tons per day based on speed in km/h and ice resistance.
        """

        self.mesh_df['fuel'] = (0.00137247 * self.mesh_df['speed'] ** 2 - 0.0029601 *
                                self.mesh_df['speed'] + 0.25290433
                                + 7.75218178e-11 * self.mesh_df['ice resistance'] ** 2
                                + 6.48113363e-06 * self.mesh_df['ice resistance']) * 24.0

    def remove_nodes(self, neighbour_graph, inaccessible_nodes):
        """
            neighbour_graph -> a dictionary containing indexes of cellboxes
            and how they are connected

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

            inaccessible_nodes -> a list in indexes to be removed from the
            neighbour_graph
        """
        accessibility_graph = neighbour_graph.copy()

        for node in accessibility_graph.keys():
            for case in accessibility_graph[node].keys():
                for inaccessible_node in inaccessible_nodes:
                    if int(inaccessible_node) in accessibility_graph[node][case]:
                        accessibility_graph[node][case].remove(int(inaccessible_node))

        for node in inaccessible_nodes:
            accessibility_graph.pop(node)

        return accessibility_graph
