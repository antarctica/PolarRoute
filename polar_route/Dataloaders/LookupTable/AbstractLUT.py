from polar_route.Dataloaders.DataLoaderInterface import DataLoaderInterface
from abc import abstractmethod

class LookupTableDataLoader(DataLoaderInterface):
    
    # User defines Boundary objects
    # Have to ensure they cover entire domain of lat/long/time
    # Dict of boundary to value
    
    # When get_value called, input bounds
    # Figure which boundaries 'bounds' overlaps
    # calculate how much area/volume bounds takes within each Boundary
    # add all together to get value
    @abstractmethod
    def __init__(self, bounds, params):
        pass
    
    @abstractmethod
    def setup_lookup_table(self):
        # Returns dict of
        # {Boundary: value}
        pass

    def get_value(self, bounds):
        pass        

    def coord_to_boundary(self, coords):
        # If coord lat < boundary_lat_min, drop
        # If coords lat > boundary_lat_max, drop
        # Same for long
        # Same for time
        # Return list of boundaries that haven't been dropped
        pass
    