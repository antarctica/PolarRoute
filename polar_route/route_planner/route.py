import numpy as np
from polar_route.utils import case_from_angle, unit_time, unit_speed


class Route:
    """
        class that represents a route from a start to end waypoints
        Attributes:

            segments (list<Segment>): a list of Segment object that constitutes the entire route
            name (String) (optional): string representing the route name
            _from (string): the name of the source waypoint
            _to (string): the name of the destination waypoint

    """

    def __init__(self, segments, _from, _to, conf, name=None):
        self.segments = segments
        if name is None:
            self.name = 'Route Path - {} to {}'.format(_from, _to)
        self._from = _from
        self._to = _to
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
        path['geometry']['coordinates'] =  self.get_points()
        path['properties'] = {}
        path['properties']['name'] = self.name
        path['properties']['from'] = self._from
        path['properties']['to'] = self._to

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
        metrics = [getattr(seg, metric) for seg in self.segments]
        return [ sum(metrics [0:i+1]) for i in range(len(metrics))]

    def set_cases(self, cases):
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
       # print ("dist in cell >>> ", dist)
        return traveltime, dist
    
    def _waypoint_correction(self, cellbox, wp, cp, indx):
        """
            Determine within cell parameters for a source and end point on the edge
        """
        # case = -1
        print(">> wp >>", wp.to_point())
        print(">> cp >>", cp.to_point())
        m_long  = 111.321*1000
        m_lat   = 111.386*1000
        # x = self._dist_around_globe(cp.get_longitude(),wp.get_longitude())*m_long*np.cos(wp.get_latitude()*(np.pi/180))
        x = (cp.get_longitude() - wp.get_longitude()) * m_long * np.cos(wp.get_latitude() * (np.pi / 180))
        y = (cp.get_latitude() - wp.get_latitude()) * m_lat
        if indx == 0:
            case = case_from_angle(wp.to_point(), cp.to_point())
        else: 
            case = case_from_angle(cp.to_point(), wp.to_point())
        su  = cellbox.agg_data['uC']
        sv  =  cellbox.agg_data['vC']
        ssp = unit_speed(cellbox.agg_data['speed'][case], self.conf['unit_shipspeed'])
        print(case, x, y, su, sv, ssp)
        traveltime, distance = self._traveltime_in_cell(x, y, su, sv, ssp)
        print("WP_correction>>> tt >> ", traveltime)
        print("WP_correction>>> distance >> ", distance)
        # traveltime = unit_time(traveltime, self.conf['time_unit'])
        print("case >> ", case)

        # update segment and its metrics
     
        self.segments[indx].set_waypoint(indx, wp)
        self.segments[indx].set_travel_time(traveltime)
        self.segments[indx].set_distance(distance)
        self.segments[indx].set_fuel(cellbox.agg_data['fuel'][case] * traveltime)

    def _dist_around_globe(self, sp, cp):
        a1 = np.sign(cp-sp)*(np.max([sp,cp])-np.min([sp,cp]))
        a2 = -(360-(np.max([sp,cp])-np.min([sp,cp])))*np.sign(cp-sp)

        dist = [a1, a2]
        indx = np.argmin(abs(np.array(dist)))

        a = dist[indx]
        return a   

    def get_points(self):
        points = []
        if len(self.segments) > 0:
            points.append(self.segments[0].get_start_wp().to_point())

            for seg in self.segments:
                 points.append(seg.get_end_wp().to_point())
        return points
