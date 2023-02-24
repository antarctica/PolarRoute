from polar_route.Dataloaders.DataLoaderInterface import DataLoaderInterface
from abc import abstractmethod
from polar_route.utils import rectangle_overlap, str_to_datetime, frac_of_month, boundary_to_coords
import numpy as np
from pandas import date_range
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
    def import_data(self):
        '''
        Reads in JSON file to set up boundaries
        '''
        pass

    def get_value(self, bounds):
        
        # Get list of every month in time range of bounds
        months_in_bounds = [
            (int(d.strftime("%Y")), int(d.strftime("%m"))) \
            for d in date_range(
                start=bounds.get_time_min(), 
                end=bounds.get_time_max(), 
                freq='MS')
            ]

        # Determine area of bounds
        b_coords = boundary_to_coords(bounds)
        bounds_area = rectangle_overlap(b_coords, b_coords)
        # Convert boundary time boundaries to datetime objects
        min_dt = str_to_datetime(bounds.get_time_min())
        max_dt = str_to_datetime(bounds.get_time_max())
        # Initialise empty arrays to populate
        monthly_values = np.array([])
        monthly_value_weights = np.array([])
        # For each month
        for year, month in months_in_bounds:
            # Reset total cumulative area, bounds value
            cumulative_area_frac = 0
            bounds_value = 0
            # For each area defined by user
            for key, area in self.data.items():
                # Extract the boundary
                area_boundary = area['boundary']
                # If it overlaps 'bounds' at all
                # Or is to accept when bounds larger than area and vice versa
                if self.bounds_in_boundary(area_boundary, bounds):
                    # Set up coordinates of extreme points
                    a_coords = boundary_to_coords(area_boundary)
                    # Determine how much area this accounts for in total bounds
                    overlaped_area = rectangle_overlap(a_coords, b_coords)
                    # Save as a fraction of total area
                    area_fraction = overlaped_area/bounds_area
                    # Add towards cumulative area found
                    cumulative_area_frac += area_fraction
                    # Extract weighted LUT value for this season
                    # Add to total value within bounds
                    bounds_value += area['value'] * area_fraction

                    print(key)
                    # TODO Add time to fractional area/'volume'
            # Assert all areas added up account for all of bounds
            assert (cumulative_area_frac == 1), \
                f'Only have data for {cumulative_area_frac*100}% of boundary!'
            
            # Determine how much of the month the temporal bounds count for
            # If start/end not on start/end of month, then less weight given
            if month == min_dt.month and year == min_dt.year:
                monthly_frac = frac_of_month(year, month, start_date=min_dt)
            elif month == max_dt.month and year == max_dt.year:
                monthly_frac = frac_of_month(year, month, end_date=max_dt)
            else:
                monthly_frac = 1
            # Save each month's value in a list
            monthly_values = np.append(monthly_values, bounds_value)
            monthly_value_weights = np.append(monthly_value_weights, monthly_frac)            
        # Normalise weights of months
        monthly_value_weights = monthly_value_weights / np.sum(monthly_value_weights)
        
        # Weight values and add together
        final_value = np.sum(monthly_values * monthly_value_weights)
        
        return final_value
    
    def get_hom_condition(self, bounds, splitting_conditions):
        # Retrieve boundaries to analyse
        areas = [area for area in self.data 
                if self.bounds_in_boundary(area['boundary'], bounds)]
        
        # If no area specified by bounds
        if len(areas) == 0: return 'MIN'
        # Otherwise, extract the homogeneity condition

        # Calculate fraction over threshold
        for area in areas:
            if area.value > splitting_conditions['threshold']:
                area_coords = boundary_to_coords(area['boundary'])
                overlap_area = rectangle_overlap(area_coords, bounds)
        
        
        num_over_threshold = dps[dps > splitting_conds['threshold']]
        frac_over_threshold = num_over_threshold.shape[0]/dps.shape[0]

        # Return homogeneity condition
        if   frac_over_threshold <= splitting_conds['lower_bound']: return 'CLR'
        elif frac_over_threshold >= splitting_conds['upper_bound']: return 'HOM'
        else: return 'HET'

    def coords_in_boundary(self, coords, bounds):
        
        if bounds.get_lat_min() <= coords[0] < bounds.get_lat_max() and \
           bounds.get_long_min()<= coords[1]< bounds.get_long_max():
            return True
        else:
            return False
        
    def bounds_in_boundary(self, local_bounds, global_boundary):
        lats = [local_bounds.get_lat_min(), local_bounds.get_lat_max()]
        longs = [local_bounds.get_long_min(), local_bounds.get_long_max()]
        local_coords = [(i,j) for i in lats
                              for j in longs]
        for coords in local_coords:
            if self.coords_in_boundary(coords, global_boundary):
                return True
        
        lats = [global_boundary.get_lat_min(), global_boundary.get_lat_max()]
        longs = [global_boundary.get_long_min(), global_boundary.get_long_max()]
        global_coords = [(i,j) for i in lats
                               for j in longs]
        
        for coords in global_coords:
            if self.coords_in_boundary(coords, local_bounds):
                return True
            
        return False

    
    def downsample(self):
        raise NotImplementedError(
            "downsample() method doesn't make sense for LUT Dataloader!"
            )
    def get_datapoints(self):
        raise NotImplementedError(
            "get_datapoints() method doesn't make sense for LUT Dataloader!"
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