from polar_route import segment


class Route:
    '''
        class that represents a route from a start to end waypoints
        Attributes:

            segments (list<Segment>): a list of Segment object that constitutes the entire route
            name (String) (optional): s tring reoresnting the route name
            _from (string): the name of the source waypoint
            _to (string): the name of the destination waypoint
            
    '''

    def __init__(self , segments, _from, _to , name = None):
        self.segments = segments
        if name == None:
            self.name = 'Route Path - {} to {}'.format(_from, _to)
        self._from = _from
        self._to = _to
        self.cases = []


    
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
      
    def to_geojson(self ):
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
        path['properties']['speed'] = [ segment.get_speed() for segment in path ]
        path['properties']['fuel'] = [ segment.get_fuel() for segment in path ]

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