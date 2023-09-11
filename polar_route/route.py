from polar_route import segment
import numpy as np
from polar_route.utils import case_from_angle, unit_time, unit_speed

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
      
    def to_json(self ):
        '''
            puts the constructed route in geojson format
        '''
        output = {}
        paths = []
        geojson = {}
        geojson['type'] = "FeatureCollection"
        paths = []
        path_variables = self.conf['path_variables']
        path = dict()
        path['type'] = "Feature"
        path['geometry'] = {}
        path['geometry']['type'] = "LineString"    
        path['geometry']['coordinates'] =  self.get_points()
        path['properties'] = {}
        path['properties']['name'] = self.name
        path['properties']['from'] = self._from
        path['properties']['to'] = self._to

        cell_indices  = []
        for segment in self.segments:
            cell_indices.append (segment.start_wp.get_cellbox_indx())
           # cell_indices.append (segment.end_wp.get_cellbox_indx())
        # path_indices = np.array([cellIndices[0]] + list(np.repeat(cellIndices[1:-1], 2)) + [cellIndices[-1]]) ???
        path['properties']['CellIndices'] = cell_indices
        path['properties']['traveltime']  = self._accumulate_metric ('traveltime')
        path['properties']['cases'] = self.cases
        for variable in path_variables: 
             path['properties'][variable] = self._accumulate_metric (variable) 

        paths.append(path)
        geojson['features'] = paths
        output['paths'] = geojson
        return output

    def _accumulate_metric (self, metric):
        metrics = [getattr(segment, metric)  for segment in self.segments]
        return [ sum(metrics [0:i+1]) for i in range (len(metrics))]

           


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
       # print ("dist in cell >>> " , dist)
        return traveltime, dist
    
    def _waypoint_correction(self,cellbox,wp, cp , indx):
        '''
            Determine within cell parameters for a source and end point on the edge
        '''
        
        case = -1
        print (">> wp >>", wp.to_point())
        print (">> cp >>" ,cp.to_point())
        m_long  = 111.321*1000
        m_lat   = 111.386*1000
        # x = self._dist_around_globe(cp.get_longtitude(),wp.get_longtitude())*m_long*np.cos(wp.get_latitude()*(np.pi/180))
        x =  (cp.get_longtitude()-wp.get_longtitude())*m_long*np.cos(wp.get_latitude()*(np.pi/180))
        y = (cp.get_latitude()-wp.get_latitude())*m_lat
        if indx == 0:
            case = case_from_angle(wp.to_point(),cp.to_point())
        else: 
            case = case_from_angle(cp.to_point(),wp.to_point())
        Su  = cellbox.agg_data['uC']
        Sv  =  cellbox.agg_data['vC']
        Ssp = unit_speed(cellbox.agg_data['speed'][case] , self.conf['unit_shipspeed'])
        print (case, x,y,Su,Sv,Ssp)
        traveltime, distance = self._traveltime_in_cell(x,y,Su,Sv,Ssp)
        print ("WP_correction>>> tt >> " , traveltime)
        print ("WP_correction>>> distance >> " , distance)
        traveltime = unit_time(traveltime , self.conf['time_unit'])
        print ("case >> " , case)
       

        # update segment and its metrics
     
        self.segments[indx].set_waypoint (indx , wp)
        self.segments[indx].set_travel_time (traveltime)
        self.segments[indx].set_distance (distance)
        self.segments[indx].set_fuel (cellbox.agg_data['fuel'] [case] * traveltime)
        

    def _dist_around_globe( Sp,Cp):
        a1 = np.sign(Cp-Sp)*(np.max([Sp,Cp])-np.min([Sp,Cp]))
        a2 = -(360-(np.max([Sp,Cp])-np.min([Sp,Cp])))*np.sign(Cp-Sp)

        dist = [a1,a2]
        indx = np.argmin(abs(np.array(dist)))

        a = dist[indx]
        return a   
    

    
    def get_points(self):
        points = []
        if len(self.segments) > 0:
            points.append (self.segments[0].get_start_wp().to_point())

            for segment in self.segments:
                 points.append (segment.get_end_wp().to_point())
        return points
        
