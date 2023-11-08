from meshiphi.mesh_generation.aggregated_cellbox import AggregatedCellBox
from polar_route.vessel_performance.vessels.abstract_ship import AbstractShip
import numpy as np
import logging
import tensorflow as tf
import pickle
import os

class SDA_RNN(AbstractShip):
    """
        Vessel class with methods specifically designed to model the performance of the British Antarctic Survey
        research and supply ship, the RRS Sir David Attenborough (SDA)
    """
    def __init__(self, params):
        """
            Args:
                params (dict): vessel parameters from the vessel config file
        """
        super().__init__(params)

        self.force_limit = self.vessel_params['force_limit']
        self.beam = self.vessel_params['beam']
        self.hull_type = self.vessel_params['hull_type']

        self.base_path = os.path.dirname(os.path.realpath(__file__))

        # Load the trained models
        sog_model_path = os.path.join(self.base_path, 'sog_relativedir_190923.h5')
        rpm_model_path = os.path.join(self.base_path, 'power_relativedir.h5')
        spline_model_path = os.path.join(self.base_path, 'spline_model.pkl')

        self.sog_model = tf.keras.models.load_model(sog_model_path)
        self.rpm_model = tf.keras.models.load_model(rpm_model_path)
        with open(spline_model_path, "rb") as f:
            self.spline_model = pickle.load(f)

    def model_speed(self, cellbox):
        """
            Method to determine the maximum speed that the SDA can traverse the given cell

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with speed values
        """

        logging.debug(f"Calculating new speed for cellbox {cellbox.id} based on SDA ML model")

        speeds = list()
        heads = [45., 90., 135., 180., 225., 270., 315., 0.]

        # Load data for RNN from cellbox
        curr_speed = np.sqrt(cellbox.agg_data['uC']**2 + cellbox.agg_data['vC']**2)
        wind_speed = cellbox.agg_data['wind_mag']
        sigwaveh = cellbox.agg_data['swh']
        maxwaveh = cellbox.agg_data['hmax']
        wave_period = cellbox.agg_data['mwp']
        curr_dir = np.mod(180 + (180/np.pi)*np.arctan2(cellbox.agg_data['vC'], cellbox.agg_data['uC']), 360)
        wind_dir = cellbox.agg_data['wind_dir']
        wave_dir = cellbox.agg_data['mwd']

        for head in heads:
            if curr_dir > head:
                rel_curr_dir = curr_dir - head
            else:
                rel_curr_dir = 360 + (curr_dir - head)
            if wind_dir > head:
                rel_wind_dir = wind_dir - head
            else:
                rel_wind_dir = 360 + (wind_dir - head)
            if wave_dir > head:
                rel_wave_dir = wave_dir - head
            else:
                rel_wave_dir = 360 + (wave_dir - head)

            cell_vals = np.array([curr_speed, wind_speed, sigwaveh, maxwaveh, wave_period, rel_curr_dir/360., rel_wind_dir/360.,
                          rel_wave_dir/360.])

            input_data = cell_vals.reshape(1, 1, cell_vals.shape[0])

            output = self.sog_model.predict(input_data)

            speed = output[0][0] * 1.852

            speeds.append(speed)

        cellbox.agg_data['speed'] = speeds

        return cellbox

    def model_power(self, cellbox):
        """
            Method to determine the power requirement of the SDA when traversing given cell

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with power values
        """

        logging.debug(f"Calculating power requirement in kW for cellbox {cellbox.id} based on SDA ML model")

        powers = list()
        heads = [45., 90., 135., 180., 225., 270., 315., 0.]

        # Load data for rnn from cellbox
        curr_speed = np.sqrt(cellbox.agg_data['uC'] ** 2 + cellbox.agg_data['vC'] ** 2)
        wind_speed = cellbox.agg_data['wind_mag']
        sigwaveh = cellbox.agg_data['swh']
        maxwaveh = cellbox.agg_data['hmax']
        wave_period = cellbox.agg_data['mwp']
        curr_dir = np.mod(180 + (180 / np.pi) * np.arctan2(cellbox.agg_data['vC'], cellbox.agg_data['uC']), 360)
        wind_dir = cellbox.agg_data['wind_dir']
        wave_dir = cellbox.agg_data['mwd']

        for head in heads:
            if curr_dir > head:
                rel_curr_dir = curr_dir - head
            else:
                rel_curr_dir = 360 + (curr_dir - head)
            if wind_dir > head:
                rel_wind_dir = wind_dir - head
            else:
                rel_wind_dir = 360 + (wind_dir - head)
            if wave_dir > head:
                rel_wave_dir = wave_dir - head
            else:
                rel_wave_dir = 360 + (wave_dir - head)

            cell_vals = np.array([curr_speed, wind_speed, sigwaveh, maxwaveh, wave_period, rel_curr_dir/360., rel_wind_dir/360.,
                          rel_wave_dir/360.])

            input_data = cell_vals.reshape(1, 1, cell_vals.shape[0])

            rpm = self.rpm_model.predict(input_data)

            power = self.spline_model.predict(rpm)

            powers.append(power[0][0])

        cellbox.agg_data['power'] = powers

        return cellbox


    def model_fuel(self, cellbox):
        """
            Method to determine the fuel consumption rate of the SDA in a given cell

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with fuel consumption values
        """
        logging.debug(f"Calculating fuel requirements in cell {cellbox.id}")

        cellbox = self.model_resistance(cellbox)

        cellbox.agg_data['fuel'] = [fuel_eq(cellbox.agg_data['speed'][i], r)
                                    for i, r in enumerate(cellbox.agg_data['resistance'])]
        return cellbox


    def model_resistance(self, cellbox):
        """
            Method to determine the resistance force acting on the SDA in a given cell

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with resistance values
        """

        # Include ice resistance if present
        if 'ice resistance' in cellbox.agg_data:
            ice_resistance = [cellbox.agg_data['ice resistance'] for x in range(8)]
        else:
            ice_resistance = [0, 0, 0, 0, 0, 0, 0, 0]

        # Calculate wind resistance if wind data present:
        if 'u10' in cellbox.agg_data and 'v10' in cellbox.agg_data:
            cellbox = calc_wind(cellbox)
            cellbox.agg_data['resistance'] = [cellbox.agg_data['wind resistance'][i] + ice_resistance[i] for i in range(8)]
        else:
            logging.debug("No wind data present, wind resistance will not be calculated")
            cellbox.agg_data['resistance'] = ice_resistance

        return cellbox

    def ice_resistance(self, cellbox):
        """
            Method to find the ice resistance force acting on the SDA at a given speed in a given cell

            The input cellbox should contain the following values:
                velocity (float): The speed of the vessel in km/h
                sic (float): The average sea ice concentration in the cell as a percentage
                thickness (float): The average ice thickness in the cell in m
                density (float): The average ice density in the cell in kg/m^3

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                resistance (float): Resistance force in N
        """

        velocity = cellbox.agg_data['speed']
        sic = cellbox.agg_data['SIC']
        thickness = cellbox.agg_data['thickness']
        density = cellbox.agg_data['density']

        # If there's no ice then return zero
        if not sic:
            return 0.

        # Model parameters for different hull types
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

        hull = self.hull_type
        beam = self.beam
        kparam, bparam, nparam = hull_params[hull]
        gravity = 9.81  # m/s-2

        speed = velocity * (5. / 18.)  # assume km/h and convert to m/s

        froude = speed / np.sqrt(gravity * (sic / 100) * thickness)
        resistance = 0.5 * kparam * (froude ** bparam) * density * beam * thickness * (speed ** 2) * (
                    (sic / 100) ** nparam)
        return resistance

    def invert_resistance(self, cellbox):
        """
            Method to find the vessel speed that keeps the ice resistance force below a given threshold in a given cell

            The input cellbox should contain the following values\n
                sic (float) - The average sea ice concentration in the cell as a percentage \n
                thickness (float) - The average ice thickness in the cell in m \n
                density (float) - The average ice density in the cell in kg/m^3 \n

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                new_speed (float): Safe vessel speed in km/h
        """

        sic = cellbox.agg_data['SIC']
        thickness = cellbox.agg_data['thickness']
        density = cellbox.agg_data['density']

        # If there's no ice then return max speed
        if not sic:
            return self.max_speed

        # Model parameters for different hull types
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}
        # force_limit (float): Resistance force value that should not be exceeded in N
        force_limit = self.force_limit

        hull = self.hull_type
        beam = self.beam
        kparam, bparam, nparam = hull_params[hull]
        gravity = 9.81  # m/s-2

        vexp = 2 * force_limit / (kparam * density * beam * thickness * ((sic / 100) ** nparam) * (
                gravity * thickness * sic / 100) ** -(bparam / 2))

        vms = vexp ** (1 / (2.0 + bparam))
        new_speed = vms * (18. / 5.)  # convert from m/s to km/h

        return new_speed

    def wave_resistance(self, w_height):
        """
        Method to calculate the wave resistance given the wave height and vessel geometry.
        Recommended by the ITTC for small wave heights: https://ittc.info/media/1936/75-04-01-012.pdf
        """
        rho_w = 9807 # N/m^3 (specific weight of water at 4°C from wikipedia, will vary with temp and salinity)
        beam = self.vessel_params['Beam']
        c_block = self.vessel_params.get('c_block', 0.75) # ratio of underwater volume to cuboid
        length = self.vessel_params['Length']

        wave_res = (0.64*rho_w*c_block*(w_height**2)*(beam**2))/length # Kreitner, valid up to ~2m wave height

        return wave_res


def fuel_eq(speed, resistance):
    """
        Equation to calculate the fuel consumption in tons/day given the speed in km/h and the resistance force in N

        Args:
            speed (float): the SDA's speed in km/h
            resistance (float): the resistance force in N

        Returns:
            fuel (float): the fuel consumption in tons/day
    """
    # Assume no effect from "negative resistance" (e.g. when tailwind is stronger than ice resistance)
    if resistance < 0:
        resistance = 0.
    fuel = (0.00137247 * speed ** 2 - 0.0029601 * speed + 0.25290433
          + 7.75218178e-11 * resistance ** 2 + 6.48113363e-06 * resistance) * 24.0
    return fuel


def c_wind(rel_ang):
    """
        Function to return the wind resistance coefficient for some relative angle between wind and travel directions.
    """
    cs = -1 * np.array([-0.94, -0.77, -0.42, -0.48, -0.17, 0.30, 0.34])
    angles = np.array([0., np.pi / 6., np.pi / 3., np.pi / 2., 2 * np.pi / 3., 5 * np.pi / 6., np.pi])
    return np.interp(rel_ang, angles, cs)


def wind_mag_dir(cellbox, va):
    """
        Function that returns the relative wind speed and direction given the speed and heading of the vessel
        and the easterly and northerly components of the true wind vector. Speeds in m/s.
    """
    vs = cellbox.agg_data['speed'][0]*0.277778
    uw = cellbox.agg_data['u10']
    vw = cellbox.agg_data['v10']

    # Define wind vector
    w_vec = np.array([uw, vw])

    # Find vessel vector in component form
    uv, vv = vs * np.sin(va), vs * np.cos(va)
    v_vec = np.array([uv, vv])

    # Find apparent wind vector
    aw_vec = w_vec - v_vec

    # Find magnitude of apparent wind
    ws = np.linalg.norm(aw_vec)

    # Define unit vectors for dot product
    if vs:
        unit_v = v_vec / vs
    else:
        unit_v = [0., 0.]
    if ws:
        unit_aw = aw_vec / ws
    else:
        return 0., 0.

    # Calculate dot product and find angle
    dp = np.dot(unit_v, unit_aw)
    ang = np.arccos(-1 * dp)

    return ws, ang


def wind_resistance(v_speed, w_speed, rel_ang):
    """
        Function to calculate the wind resistance given the wind speed and direction and the vessel speed.
    """
    a = 750.
    rho = 1.225

    wind_res = 0.5 * rho * (w_speed**2) * a * c_wind(rel_ang) - 0.5 * rho * (v_speed ** 2) * a * c_wind(0)

    return wind_res


def calc_wind(cellbox):
    """
        Function to calculate the wind resistance as well as the relative wind speed and angle for a vessel traversing a
        cell with 8 different equally spaced angular headings.

        Args:
            cellbox (AggregatedCellBox): input cell from environmental mesh

        Returns:
            cellbox (AggregatedCellBox): updated cell with wind information
    """

    logging.debug(f"Calculating wind resistance in cellbox {cellbox.id}")

    wind_res = [0, 0, 0, 0, 0, 0, 0, 0]
    rel_wind_speed = [0, 0, 0, 0, 0, 0, 0, 0]
    rel_wind_angle = [0, 0, 0, 0, 0, 0, 0, 0]

    heads = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 0.]
    for i, head in enumerate(heads):
        rel_wind_speed[i], rel_wind_angle[i] = wind_mag_dir(cellbox, head)
        wind_res[i] = wind_resistance(cellbox.agg_data['speed'][i]*0.277778, rel_wind_speed[i],
                                           rel_wind_angle[i]) # assume km/h and convert to m/s

    cellbox.agg_data['wind resistance'] = wind_res
    cellbox.agg_data['relative wind speed'] = rel_wind_speed
    cellbox.agg_data['relative wind angle'] = rel_wind_angle

    return cellbox




