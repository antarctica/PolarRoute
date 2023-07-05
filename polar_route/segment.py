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
        self.travel_time = np.inf
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
        self.travel_time = tt

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
        return self.travel_time  
        
    def get_fuel(self):
        '''
            returning the fuel
        '''
        return self.fuel 
        