import numpy as np

class Segment:
    '''
        class that represents a portion of the route from a start to end waypoints and encapsulates metrics related to this portion(ex. distance, travel time, fuel)

        Attributes:

            start_wp (Waypoint): a waypoint object represnts that start location of this segment
            end_wp (Waypoint): a waypoint object represents the end location of this segment
            distance (float): the distance between segment's start and end
            travel_time (float): the time needed to travel from segment's start to end
            fuel (float): the amount of fuel needed to travelf from segment's start to end
            
    '''

    def __init__(self , start_wp, end_wp):
        self.start_wp = start_wp
        self.end_wp = end_wp
        self.distance = np.inf
        self.traveltime = np.inf
        self.fuel = np.inf

    def set_distance (self, distance):
        '''
            setting the distance
        '''
        self.distance = distance

    def set_travel_time (self, tt):
        '''
            setting the travel time
        '''
        self.traveltime = tt

    def set_fuel (self, fuel):
        '''
            setting the fuel
        '''
        self.fuel = fuel

    def get_distance (self):
        '''
            returning the distance
        '''
        return self.distance
    
    def get_travel_time(self):
        '''
            returning the travel_time
        '''
        return self.traveltime  
        
    def get_fuel(self):
        '''
            returning the fuel
        '''
        return self.fuel 
    
    def get_points (self):
        '''
        returns the points of the segments
        '''
        return[[self.start_wp.get_latitude(), self.start_wp.get_longtitude()] , [self.end_wp.get_latitude(), self.end_wp.get_longtitude()]]
    
    def get_variable (self, variable):
        return getattr(self, variable)
    def get_start_wp (self):
        return self.start_wp
    def get_end_wp (self):
        return self.end_wp
    
    def set_start_wp (self , wp):
         self.start_wp = wp

    def set_end_wp (self , wp):
         self.end_wp = wp
    def set_waypoint (self, indx , wp):
        if indx ==0: #set the start waypoint
            self.set_start_wp(wp)
        elif indx == -1:
            self.set_end_wp(wp)  #set the end waypoint

    def get_waypoint (self, indx ):
        if indx ==0: #get the start waypoint
            return self.start_wp
        elif indx == -1:
            return self.end_wp #get the end waypoint