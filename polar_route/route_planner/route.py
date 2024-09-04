import logging
import json
import numpy as np
import geopandas as gpd
from polar_route.route_planner.crossing import traveltime_in_cell
from polar_route.utils import unit_time, unit_speed, case_from_angle
from meshiphi.utils import longitude_domain


class Route:
    """
        Class that represents a route from a start to an end waypoint
        Attributes:
            segments (list<Segment>): a list of Segment object that constitutes the entire route
            name (String) (optional): string representing the route name
            from_wp (string): the name of the source waypoint
            to_wp (string): the name of the destination waypoint
            conf (dict): the associated route config
            cases (list): a list of all cases along the route

    """

    def __init__(self, segments, _from, _to, conf, name=None):
        self.segments = segments
        if name is None:
            self.name = 'Route - {} to {}'.format(_from, _to)
        self.from_wp = _from
        self.to_wp = _to
        self.cases = []
        self.conf = conf
        self.source_waypoint = None
    
    def get_distance(self):
        """
            Goes through the route segments and calculates the total route distance
        """
        return sum(seg.get_distance() for seg in self.segments)
    
    def get_time(self):
        """
            Goes through the route segments and calculates the total travel time of the route
        """
        return sum(seg.get_travel_time() for seg in self.segments)
    
    def get_fuel(self):
        """
            Goes through the route segments and calculates the entire route's fuel usage
        """
        return sum(seg.get_fuel() for seg in self.segments)

    def get_battery(self):
        """
            Goes through the route segments and calculates the entire route's battery usage
        """
        return sum(seg.get_battery() for seg in self.segments)

    def accumulate_metric(self, metric):
        """
        Finds the cumulative sum over the given metric for the route
        Args:
            metric (str): The name of the metric

        Returns:
            metric_cumul (list): List of cumulative values for the metric at each segment along the route
        """
        metrics = [getattr(seg, metric) for seg in self.segments]
        metric_cumul = [sum(metrics[0:i+1]) for i in range(len(metrics))]
        return metric_cumul

    def set_cases(self, cases):
        """
            Sets the cases attribute for the route to the given value
        """
        self.cases = cases
      
    def to_json(self):
        """
            Converts the constructed route into json format
        """
        route = dict()

        path_variables = self.conf['path_variables']
        route['type'] = "Feature"
        route['geometry'] = {}
        route['geometry']['type'] = "LineString"
        route['geometry']['coordinates'] = longitude_domain(self.get_points())
        route['properties'] = {}
        route['properties']['name'] = self.name
        route['properties']['from'] = self.from_wp
        route['properties']['to'] = self.to_wp

        cell_indices = []
        for seg in self.segments:
            cell_indices.append(seg.start_wp.get_cellbox_indx())

        route['properties']['CellIndices'] = cell_indices
        route['properties']['traveltime'] = self.accumulate_metric('traveltime')
        route['properties']['cases'] = self.cases
        for variable in path_variables: 
             route['properties'][variable] = self.accumulate_metric(variable)
             route['properties']['total_' + variable] = route['properties'][variable][-1]

        route['properties']['total_traveltime'] = route['properties']['traveltime'][-1]

        return route

    def to_geojson(self):
        """
            Converts the constructed route into geojson format
        """
        geojson = {"type": "FeatureCollection", "features": []}
        geojson['features'].append(self.to_json())

        return geojson

    def to_csv(self):
        """
           Converts the constructed route into csv format
        """
        # TODO Decide on an output format for this method or remove
        raise NotImplementedError

    def to_gpx(self):
        """
           Converts the constructed route into a geo-dataframe with fields matching gpx format
        """
        route = self.to_json()
        gdf = gpd.GeoDataFrame.from_features([route])
        return gdf

    def save(self, file_path):
        """
           Saves the constructed route to the given file location in the given format
        """
        logging.info(f"Saving route to {file_path}")
        file_path_strs = file_path.split('.')

        if file_path_strs[-1] in ["json", "geojson"]:
            with open(file_path, 'w') as f:
                json.dump(self.to_geojson(), f)
        elif file_path_strs[-1] == 'gpx':
            gdf = self.to_gpx()
            gdf['geometry'].to_file(file_path, "GPX")
        else:
            raise ValueError('Provide a path with a supported file extension: "json", "gpx"')

    def waypoint_correction(self, cellbox, wp, cp, indx):
        """
            Determine within cell parameters for the source and end point waypoint when away from cell centre and
            update the relevant segment

            Args:
                cellbox (AggregatedCellBox): the cellbox to do the waypoint correction within
                wp (Waypoint): the source or end waypoint
                cp (Waypoint): the crossing point that the route enters or leaves the cell by
                indx (int): the index of the segment along the route
        """
        direction = [1, 2, 3, 4, -1, -2, -3, -4]
        logging.debug(f"WP_correction >> wp >> {wp.to_point()}")
        logging.debug(f"WP_correction >> cp >> {cp.to_point()}")
        m_long = 111.321*1000
        m_lat = 111.386*1000
        x = (cp.get_longitude() - wp.get_longitude()) * m_long * np.cos(wp.get_latitude() * (np.pi / 180))
        y = (cp.get_latitude() - wp.get_latitude()) * m_lat
        # Select case with matching index or use angle if waypoints are in the same cell
        if self.cases:
            case = self.cases[indx]
        else:
            case = case_from_angle(wp.to_point(), cp.to_point())
        su = cellbox.agg_data['uC']
        sv = cellbox.agg_data['vC']
        ssp = unit_speed(cellbox.agg_data['speed'][case], self.conf['unit_shipspeed'])
        traveltime, distance = traveltime_in_cell(x, y, su, sv, ssp, tt_dist=True)
        logging.debug(f"WP_correction >> tt >> {traveltime}")
        logging.debug(f"WP_correction >> distance >> {distance}")
        logging.debug(f"WP_correction >> case >> {case}")
        traveltime = unit_time(traveltime, self.conf['time_unit'])

        # update segment and its metrics
        self.segments[indx].set_waypoint(indx, wp)
        self.segments[indx].set_travel_time(traveltime)
        self.segments[indx].set_distance(distance)
        if 'fuel' in self.conf['path_variables']:
            self.segments[indx].set_fuel(cellbox.agg_data['fuel'][direction.index(case)] * traveltime)
            logging.debug(f"WP_correction >> fuel >> {cellbox.agg_data['fuel'][direction.index(case)] * traveltime}")
        if 'battery' in self.conf['path_variables']:
            self.segments[indx].set_fuel(cellbox.agg_data['battery'][direction.index(case)] * traveltime)
            logging.debug(f"WP_correction >> battery >> {cellbox.agg_data['battery'][direction.index(case)] * traveltime}")

    def get_points(self):
        """
            Gets a list of points along the route
        """
        points = []
        if len(self.segments) > 0:
            points.append(self.segments[0].get_start_wp().to_point())

            for seg in self.segments:
                 points.append(seg.get_end_wp().to_point())
        return points
