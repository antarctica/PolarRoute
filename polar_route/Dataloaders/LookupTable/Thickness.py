from polar_route.Dataloaders.LookupTable.AbstractLUT import LookupTableDataLoader

from polar_route.Boundary import Boundary
import numpy as np
import pandas as pd

from polar_route.utils import rectangle_overlap, str_to_datetime, frac_of_month

class ThicknessDataLoader(LookupTableDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
            
        self.setup_lookup_table()
        
    def setup_lookup_table(self):
        
        all_boundaries = {
            "Ross E S summer": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 1.32
            },
            "Ross E S autumn": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.82
            },
            "Ross E S winter": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 0.72
            },
            "Ross E S spring": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.67
            },
            "Ross E N summer": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 1.32
            },
            "Ross E N autumn": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.82
            },
            "Ross E N winter": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 0.72
            },
            "Ross E N spring": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.67
            },
            "Ross W S summer": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 1.32
            },
            "Ross W S autumn": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.82
            },
            "Ross W S winter": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 0.72
            },
            "Ross W S spring": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.67
            },
            "Ross W N summer": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 1.32
            },
            "Ross W N autumn": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.82
            },
            "Ross W N winter": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 0.72
            },
            "Ross W N spring": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.67
            },
            "Bellinghausen S summer": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 2.14
            },
            "Bellinghausen S autumn": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.79
            },
            "Bellinghausen S winter": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 0.65
            },
            "Bellinghausen S spring": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.79
            },
            "Bellinghausen N summer": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 2.14
            },
            "Bellinghausen N autumn": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.79
            },
            "Bellinghausen N winter": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 0.65
            },
            "Bellinghausen N spring": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.79
            },
            "Weddel W S summer": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 1.2
            },
            "Weddel W S autumn": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 1.38
            },
            "Weddel W S winter": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 1.33
            },
            "Weddel W S spring": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 1.33
            },
            "Weddel W N summer": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 1.2
            },
            "Weddel W N autumn": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 1.38
            },
            "Weddel W N winter": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 1.33
            },
            "Weddel W N spring": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 1.33
            },
            "Weddel E S summer": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 0.87
            },
            "Weddel E S autumn": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.44
            },
            "Weddel E S winter": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 0.54
            },
            "Weddel E S spring": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.89
            },
            "Weddel E N summer": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 0.87
            },
            "Weddel E N autumn": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.44
            },
            "Weddel E N winter": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 0.54
            },
            "Weddel E N spring": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.89
            },
            "Indian S summer": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 1.05
            },
            "Indian S autumn": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.45
            },
            "Indian S winter": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 0.59
            },
            "Indian S spring": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.78
            },
            "Indian N summer": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 1.05
            },
            "Indian N autumn": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.45
            },
            "Indian N winter": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 0.59
            },
            "Indian N spring": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.78
            },
            "West Pacific S summer": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 1.17
            },
            "West Pacific S autumn": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.75
            },
            "West Pacific S winter": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 0.72
            },
            "West Pacific S spring": {
                "boundary": Boundary([-90, 0], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.67
            },
            "West Pacific N summer": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-06-01', '2000-08-31']),
                "value": 1.17
            },
            "West Pacific N autumn": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-09-01', '2000-11-30']),
                "value": 0.75
            },
            "West Pacific N winter": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['1999-12-01', '2000-02-29']),
                "value": 0.72
            },
            "West Pacific N spring": {
                "boundary": Boundary([0, 90], [bounds.get_long_min(), bounds.get_long_max()], ['2000-03-01', '2000-05-31']),
                "value": 0.67
            }
        }
        self.areas = list(all_boundaries)
        self.boundaries = all_boundaries
                
    def get_value(self, bounds):
        # Choose season based on which hemisphere 
        # the majority of boundary is in
        if np.mean((bounds.get_lat_min(), bounds.get_lat_max())) < 0:    
            month_to_season = {
                1:  'summer', 2:  'summer', 12: 'summer',
                3:  'autumn', 4:  'autumn', 5:  'autumn', 
                6:  'winter', 7:  'winter', 8:  'winter',
                9:  'spring', 10: 'spring', 11: 'spring'
            }
        else:
            month_to_season = {
                1:  'winter', 2:  'winter', 12: 'winter',
                3:  'spring', 4:  'spring', 5:  'spring', 
                6:  'summer', 7:  'summer', 8:  'summer',
                9:  'autumn', 10: 'autumn', 11: 'autumn'
            }

        # Get list of every month in time range of bounds
        months_in_bounds = [
            (int(d.strftime("%Y")), int(d.strftime("%m"))) \
            for d in pd.date_range(
                start=bounds.get_time_min(), 
                end=bounds.get_time_max(), 
                freq='MS')
            ]

        # Determine area of bounds
        b_coords = ((bounds.get_lat_min(), bounds.get_long_min()),
                    (bounds.get_lat_max(), bounds.get_long_max()))
        bounds_area = rectangle_overlap(b_coords, b_coords)
        # Convert boundary time boundaries to datetime objects
        min_dt = str_to_datetime(bounds.get_time_min())
        max_dt = str_to_datetime(bounds.get_time_max())
        # Initialise empty arrays to populate
        monthly_values = np.array([])
        monthly_value_weights = np.array([])
        # For each month
        for year, month in months_in_bounds:
            # Determine which season it's in
            season = month_to_season[month]
            # Reset total cumulative area, bounds value
            cumulative_area_frac = 0
            bounds_value = 0
            # For each area defined by user
            for area in self.areas:
                # Extract the boundary
                area_boundary = self.boundaries[area]['boundary']
                # If it overlaps 'bounds' at all
                # Or is to accept when bounds larger than area and vice versa
                if self.bounds_in_boundary(area_boundary, bounds) or \
                   self.bounds_in_boundary(bounds, area_boundary):
                    # Set up coordinates of extreme points
                    a_coords = ((area_boundary.get_lat_min(), area_boundary.get_long_min()),
                                (area_boundary.get_lat_max(), area_boundary.get_long_max()))
                    # Determine how much area this accounts for in total bounds
                    overlaped_area = rectangle_overlap(a_coords, b_coords)
                    # Save as a fraction of total area
                    area_fraction = overlaped_area/bounds_area
                    # Add towards cumulative area found
                    cumulative_area_frac += area_fraction
                    # Extract LUT value for this season
                    lut_value = self.boundaries[area]['value'][season]
                    # Add to total value within bounds
                    bounds_value += lut_value * area_fraction
            
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
    

if __name__ == '__main__':
    bounds = Boundary(
        [20, 40],
        [-70,-50],
        ['2020-02-01', '2020-03-31']
    )
    params = {
        'data_name': 'thickness'
    }
    tdl = ThicknessDataLoader(bounds, params)
    value = tdl.get_value(bounds)
    
    print(value)