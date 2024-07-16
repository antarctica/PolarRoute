import logging
import numpy as np
from polar_route.utils import case_from_angle, unit_time, unit_speed
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
            self.name = 'Route Path - {} to {}'.format(_from, _to)
        self.from_wp = _from
        self.to_wp = _to
        self.cases = []
        self.conf = conf
    
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
        return  sum(seg.get_fuel() for seg in self.segments)
      
    def to_json(self):
        """
            Converts the constructed route into geojson format
        """
        output = dict()
        geojson = dict()
        path = dict()
        paths = list()

        geojson['type'] = "FeatureCollection"
        path_variables = self.conf['path_variables']
        path['type'] = "Feature"
        path['geometry'] = {}
        path['geometry']['type'] = "LineString"    
        path['geometry']['coordinates'] =  longitude_domain(self.get_points())
        path['properties'] = {}
        path['properties']['name'] = self.name
        path['properties']['from'] = self.from_wp
        path['properties']['to'] = self.to_wp

        cell_indices  = []
        for seg in self.segments:
            cell_indices.append(seg.start_wp.get_cellbox_indx())
           # cell_indices.append(segment.end_wp.get_cellbox_indx())
        # path_indices = np.array([cellIndices[0]] + list(np.repeat(cellIndices[1:-1], 2)) + [cellIndices[-1]]) ???
        path['properties']['CellIndices'] = cell_indices
        path['properties']['traveltime']  = self._accumulate_metric('traveltime')
        path['properties']['cases'] = self.cases
        for variable in path_variables: 
             path['properties'][variable] = self._accumulate_metric(variable)

        paths.append(path)
        geojson['features'] = paths
        output['paths'] = geojson
        return output

    def _accumulate_metric(self, metric):
        """
        Finds the cumulative sum over the given metric for the route
        Args:
            metric (str): The name of the metric

        Returns:
            metric_cumul (list): List of cumulative values for the metric at each segment along the route
        """
        metrics = [getattr(seg, metric) for seg in self.segments]
        metric_cumul = [sum(metrics [0:i+1]) for i in range(len(metrics))]
        return metric_cumul

    def set_cases(self, cases):
        """
            Sets the cases attribute for the route to the given value
        """
        self.cases = cases

    def to_csv(self):
        """
           Converts the constructed route into csv format
        """

    def to_gpx(self):
        """
           Converts the constructed route into gpx format
        """
    def save(self, file_path):
        """
           Saves the constructed route to the given file location
        """

    def _traveltime_in_cell(self, xdist, ydist, u, v, s):
        """
            Determine the traveltime within a cell
        """
        dist  = np.sqrt(xdist**2 + ydist**2)
        cval  = np.sqrt(u**2 + v**2)

        dotprod  = xdist*u + ydist*v
        diffsqrs = s**2 - cval**2

        # if (dotprod**2 + diffsqrs*(dist**2) < 0)
        if diffsqrs == 0.0:
            if dotprod == 0.0:
                return np.inf
                #raise Exception(' ')
            else:
                if ((dist**2)/(2*dotprod))  <0:
                    return np.inf
                    #raise Exception(' ')
                else:
                    traveltime = dist * dist / (2 * dotprod)
                    return traveltime

        traveltime = (np.sqrt(dotprod**2 + (dist**2)*diffsqrs) - dotprod)/diffsqrs
        if traveltime < 0:
            traveltime = np.inf

        return traveltime, dist
    
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

        logging.debug(f"WP_correction >> wp >> {wp.to_point()}")
        logging.debug(f"WP_correction >> cp >> {cp.to_point()}")
        m_long  = 111.321*1000
        m_lat   = 111.386*1000
        x = (cp.get_longitude() - wp.get_longitude()) * m_long * np.cos(wp.get_latitude() * (np.pi / 180))
        y = (cp.get_latitude() - wp.get_latitude()) * m_lat
        # Select case with matching index
        case = self.cases[indx]
        su  = cellbox.agg_data['uC']
        sv  =  cellbox.agg_data['vC']
        ssp = unit_speed(cellbox.agg_data['speed'][case], self.conf['unit_shipspeed'])
        traveltime, distance = self._traveltime_in_cell(x, y, su, sv, ssp)
        logging.debug(f"WP_correction >> tt >> {traveltime}")
        logging.debug(f"WP_correction >> distance >> {distance}")
        logging.debug(f"WP_correction >> case >> {case}")
        traveltime = unit_time(traveltime, self.conf['time_unit'])

        # update segment and its metrics
        self.segments[indx].set_waypoint(indx, wp)
        self.segments[indx].set_travel_time(traveltime)
        self.segments[indx].set_distance(distance)
        self.segments[indx].set_fuel(cellbox.agg_data['fuel'][case] * traveltime)
        logging.debug(f"WP_correction >> fuel >> {cellbox.agg_data['fuel'][case] * traveltime}")

    def _dist_around_globe(self, sp, cp):
        a1 = np.sign(cp-sp)*(np.max([sp,cp])-np.min([sp,cp]))
        a2 = -(360-(np.max([sp,cp])-np.min([sp,cp])))*np.sign(cp-sp)

        dist = [a1, a2]
        indx = np.argmin(abs(np.array(dist)))

        a = dist[indx]
        return a   

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
