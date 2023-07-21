class Route:
    '''
        class that represents a route from a start to end waypoints
        Attributes:

            segments (list<Segment>): a list of Segment object that constitutes the entire route
            name (String) (optional): s tring reoresnting the route name
            
    '''

    def __init__(self , segments, name= None):
        self.segments = segments
        self.name = name


     
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
           
        '''

    def to_csv(self ):
        '''
           
        '''

    def to_gpx(self ):
        '''
           
        '''
    def save (self, file_path ):
        '''
           
        '''