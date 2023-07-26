from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader

import xarray as xr

class BSOSEDepthDataLoader(ScalarDataLoader):
    def import_data(self, bounds):
        """
        Reads in data from a BSOSE Depth NetCDF file.
        Renames coordinates to 'lat' and 'long', and renames variable to
        'elevation'

        Args:
            bounds (Boundary): Initial boundary to limit the dataset to

        Returns:
            xr.Dataset:
                BSOSE Depth dataset within limits of bounds.
                Dataset has coordinates 'lat', 'long', and variable 'elevation'
        """
        # Open Dataset
        if len(self.files) == 1:    data = xr.open_dataset(self.files[0])
        else:                       data = xr.open_mfdataset(self.files)
        # Change column names
        data = data.rename({'Depth': 'elevation',
                            'YC': 'lat',
                            'XC': 'long'})
        # Change domain of dataset from [0:360) to [-180:180)
        data = data.assign_coords(long=((data.long + 180) % 360) - 180)
        # Sort the 'long' axis so that sel() will work
        data = data.sortby('long')
        # Convert elevation from coord to variable
        data = data.reset_coords('elevation')
        # Convert depth to elevation by inverting sign
        data['elevation'] = -1 * data['elevation']
        # Limit to just elevation data
        data = data['elevation'].to_dataset()
        # Trim to initial datapoints
        data = self.trim_datapoints(bounds, data=data)
        
        return data
