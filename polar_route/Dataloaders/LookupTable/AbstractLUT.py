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
        '''
        Returns:
            (dict)
        {
            'area_name':{
                'boundary': Boundary(
                    [lat_min, lat_max],
                    [long_min, long_max],
                    ['2000-01-01', '2000-12-31'] # Dates unimportant, only coords used
                    ),
                'value': {
                    'summer': int,
                    'autumn': int,
                    'winter': int,
                    'spring': int,
                }
            }
        }
        '''

        pass

    def get_value(self, bounds):
        pass      
    
    def get_hom_condition(self):
        pass  

    def coords_in_boundary(self, coords, bounds):
        
        if bounds.get_lat_min() <= coords['lat'] < bounds.get_lat_max() and \
           bounds.get_long_min()<= coords['long']< bounds.get_long_max():
            return True
        else:
            return False
        
    def bounds_in_boundary(self, local_bounds, global_boundary):
        
        bot_l = {'lat': local_bounds.get_lat_min(), 'long': local_bounds.get_long_min()}
        top_l = {'lat': local_bounds.get_lat_min(), 'long': local_bounds.get_long_max()}
        bot_r = {'lat': local_bounds.get_lat_max(), 'long': local_bounds.get_long_min()}
        top_r = {'lat': local_bounds.get_lat_max(), 'long': local_bounds.get_long_max()}
        
        if self.coords_in_boundary(bot_l, global_boundary) or \
            self.coords_in_boundary(top_l, global_boundary) or \
            self.coords_in_boundary(bot_r, global_boundary) or \
            self.coords_in_boundary(top_r, global_boundary):
            return True
        else:
            return False

    
    def downsample(self):
        raise NotImplementedError(
            "downsample() method doesn't make sense for LUT Dataloader!"
            )
    def get_datapoints(self):
        raise NotImplementedError(
            "get_datapoints() method doesn't make sense for LUT Dataloader!"
            )
    def import_data(self):
        raise NotImplementedError(
            "import_data() method doesn't make sense for LUT Dataloader!"
            )
    def reproject(self):
        raise NotImplementedError(
            "reproject() method doesn't make sense for LUT Dataloader!"
            )
    def get_data_col_name(self):
        raise NotImplementedError(
            "get_data_col_name() method doesn't make sense for LUT Dataloader!"
            )
    def set_data_col_name(self):
        raise NotImplementedError(
            "set_data_col_name() method doesn't make sense for LUT Dataloader!"
            )