import numpy as np


class Segment:
    """
        Class that represents a portion of the route from a start to an end waypoint and encapsulates metrics related
        to this portion (e.g. distance, travel time, fuel)

        Attributes:

            start_wp (Waypoint): a waypoint object that represents the start location of this segment
            end_wp (Waypoint): a waypoint object that represents the end location of this segment
            distance (float): the distance between segment's start and end waypoints
            traveltime (float): the time needed to travel from a segment's start to its end waypoint
            fuel (float): the amount of fuel needed to travel from a segment's start to its end waypoint
            battery (float): the amount of battery power needed to travel from a segment's start to its end waypoint
    """

    def __init__(self, start_wp, end_wp):
        self.start_wp = start_wp
        self.end_wp = end_wp
        self.distance = np.inf
        self.traveltime = np.inf
        self.fuel = np.inf
        self.battery = np.inf

    def set_distance(self, distance):
        """
            Setting the segment distance
        """
        self.distance = distance

    def set_travel_time(self, tt):
        """
            Setting the segment travel time
        """
        self.traveltime = tt

    def set_fuel(self, fuel):
        """
            Setting the segment fuel
        """
        self.fuel = fuel

    def set_battery(self, battery):
        """
            Setting the segment battery consumption
        """
        self.battery = battery

    def get_distance(self):
        """
            Returning the segment distance
        """
        return self.distance
    
    def get_travel_time(self):
        """
            Returning the segment travel time
        """
        return self.traveltime
        
    def get_fuel(self):
        """
            Returning the segment fuel
        """
        return self.fuel

    def get_battery(self):
        """
            Returning the segment battery consumption
        """
        return self.battery
    
    def get_points(self):
        """
            Returning the points of the segments
        """
        return[[self.start_wp.get_longitude(), self.start_wp.get_latitude()], [self.end_wp.get_longitude(), self.end_wp.get_latitude()]]
    
    def get_variable(self, variable):
        """
        Returns the value of the specified variable for the segment
        Args:
            variable (str): the name of the variable to get the value of

        Returns:
            val: the value of the input variable
        """
        val = getattr(self, variable)
        return val

    def get_start_wp(self):
        return self.start_wp

    def get_end_wp(self):
        return self.end_wp
    
    def set_start_wp(self, wp):
         self.start_wp = wp

    def set_end_wp(self, wp):
         self.end_wp = wp

    def set_waypoint(self, indx, wp):
        if indx == 0: # set the start waypoint
            self.set_start_wp(wp)
        elif indx == -1:
            self.set_end_wp(wp) # set the segment end waypoint

    def get_waypoint(self, indx):
        if indx == 0: # get the segment start waypoint
            return self.start_wp
        elif indx == -1:
            return self.end_wp # get the segment end waypoint
        
    def to_str(self):
        return (f"[ {self.start_wp.get_longitude()}, {self.start_wp.get_latitude()} ]"
                f" [{self.end_wp.get_longitude()}, {self.end_wp.get_latitude()}]")
