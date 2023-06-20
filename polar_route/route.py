class Route:
    '''
        class that represents a route from a start to end waypoints
        Attributes:

            route (list<Segment>): a list if Segment object that constitutes the entire route
            name (String) (optional): s tring reoresnting the route name
            
    '''

    def __init__(self , route, name= None):
        self.route = route
        self.name = name


     
    def get_distance(self ):
        '''
            goes through the route segments and calculates the entire route distance
        '''
        distance = -1
        return distance
    
    def get_time( self ):
        '''
            goes through the route segments and calculates the entire travel time of the route
        '''
        time = -1
        return time
    
    def get_fuel(self ):
        '''
            goes through the route segments and calculates the entire route's fuel usage
        '''
        distance = -1
        return distance
      
    def to_geojson(self ):
        '''
           
        '''

    def to_csv(self ):
        '''
           
        '''

    def to_gpx(self ):
        '''
           
        '''
    def save (self, path ):
        '''
           
        '''