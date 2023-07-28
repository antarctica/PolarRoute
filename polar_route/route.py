from polar_route import segment
import numpy as np

class Route:
    '''
        class that represents a route from a start to end waypoints
        Attributes:

            segments (list<Segment>): a list of Segment object that constitutes the entire route
            name (String) (optional): s tring reoresnting the route name
            _from (string): the name of the source waypoint
            _to (string): the name of the destination waypoint
            
    '''

    def __init__(self , segments, _from, _to , conf, name = None):
        self.segments = segments
        if name == None:
            self.name = 'Route Path - {} to {}'.format(_from, _to)
        self._from = _from
        self._to = _to
        self.cases = []
        self.conf = conf


    
    def get_distance(self ):
        '''
            goes through the route segments and calculates the entire route distance
        '''
        return sum (segment.get_distance() for segment in self.segments)
    
    def get_time( self ):
        '''
            goes through the route segments and calculates the entire travel time of the route
        '''
        return sum (segment.get_travel_time() for segment in self.segments)

    
    def get_fuel(self ):
        '''
            goes through the route segments and calculates the entire route's fuel usage
        '''
        return  sum (segment.get_fuel() for segment in self.segments)
      
    def to_geojson(self, path_variables ):
        '''
            puts the constructed route in geojson format
        '''
        paths = []
        path = dict()
        path['type'] = "Feature"
        path['geometry'] = {}
        path['geometry']['type'] = "LineString"
        path_points= []      
        path['geometry']['coordinates'] =  [segment.get_points()  for segment in path ]
        path['properties'] = {}
        path['properties']['name'] = self.name
        path['properties']['from'] = self._from
        path['properties']['to'] = self._to

        cellIndices  = [ segment.start_wp.get_cellbox_indx(), segment.end_wp.get_cellbox_indx()  for segment in path ]
        # path_indices = np.array([cellIndices[0]] + list(np.repeat(cellIndices[1:-1], 2)) + [cellIndices[-1]]) ???
        path['properties']['CellIndices'] = cellIndices
        path['properties']['traveltime']  = [ segment.get_travel_time() for segment in path ]
        path['properties']['cases'] = self.cases
        for variable in path_variables: 
             path['properties'][variable] = [ segment.get_objective(variable) for segment in path ]


        return path

    def set_cases (self, cases):
          self.cases = cases

    def to_csv(self ):
        '''
           
        '''

    def to_gpx(self ):
        '''
           
        '''
    def save (self, file_path ):
        '''
           
        '''

    def _traveltime_in_cell(self,xdist,ydist,U,V,S):
        '''
            Determine the traveltime within cell
        '''
        dist  = np.sqrt(xdist**2 + ydist**2)
        cval  = np.sqrt(U**2 + V**2)

        dotprod  = xdist*U + ydist*V
        diffsqrs = S**2 - cval**2

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
        return self._unit_time(traveltime), dist
    
    def _waypoint_correction(self,cellbox,cp, indx):
        '''
            Determine within cell parameters for a source and end point on the edge
        '''
        wp = self.segments[indx].get_start_wp()
        m_long  = 111.321*1000
        m_lat   = 111.386*1000
        x = self._dist_around_globe(cp.get_longtitude(),wp.get_longtitude(),)*m_long*np.cos(wp.get_latitude()*(np.pi/180))
        y = (cp.get_latitude(),-wp.get_latitude())*m_lat
        case = self._case_from_angle(cp.to_list(),wp.to_list())
        Su  = cellbox.agg_data['uc']
        Sv  =  cellbox.agg_data['vc']
        Ssp = self._unit_speed(cellbox.agg_data['speed'][case])
        traveltime, distance = self._traveltime_in_cell(x,y,Su,Sv,Ssp)

        # update segment and its metrics
        self.segments[indx].set_start_wp (cp)
        self.segments[indx].set_travel_time (traveltime)
        self.segments[indx].set_distance (distance)
        self.segments[indx].set_speed (cellbox.agg_data['speed'] [case] * traveltime)
        self.segments[indx].set_fuel (cellbox.agg_data['fuel'] [case] * traveltime)
        

    def _dist_around_globe(Sp,Cp):
        a1 = np.sign(Cp-Sp)*(np.max([Sp,Cp])-np.min([Sp,Cp]))
        a2 = -(360-(np.max([Sp,Cp])-np.min([Sp,Cp])))*np.sign(Cp-Sp)

        dist = [a1,a2]
        indx = np.argmin(abs(np.array(dist)))

        a = dist[indx]
        return a   
    
    def _case_from_angle(self,start,end):
        """
            Determine the direction of travel between two points in the same cell and return the associated case

            Args:
                start (list): the coordinates of the start point within the cell
                end (list):  the coordinates of the end point within the cell

            Returns:
                case (int): the case to use to select variable values from a list
        """

        direct_vec = [end[0]-start[0], end[1]-start[1]]
        direct_ang = np.degrees(np.arctan2(direct_vec[0], direct_vec[1]))

        case = None

        if -22.5 <= direct_ang < 22.5:
            case = -4
        elif 22.5 <= direct_ang < 67.5:
            case = 1
        elif 67.5 <= direct_ang < 112.5:
            case = 2
        elif 112.5 <= direct_ang < 157.5:
            case = 3
        elif 157.5 <= abs(direct_ang) <= 180:
            case = 4
        elif -67.5 <= direct_ang < -22.5:
            case = -3
        elif -112.5 <= direct_ang < -67.5:
            case = -2
        elif -157.5 <= direct_ang < -112.5:
            case = -1

        return case

    def _unit_time(self,val):
        '''
            Applying Unit time for a specific input type
        '''
        if self.conf.unit_time == 'days':
            return val/(60*60*24)
        elif self.conf.unit_time == 'hr':
            return val/(60*60)
        elif self.conf.unit_time == 'min':
            return val/(60)
        elif self.conf.unit_time == 's':
            return val
    
