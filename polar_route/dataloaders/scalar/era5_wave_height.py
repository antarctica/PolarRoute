from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader

import xarray as xr


class ERA5WaveHeightDataLoader(ScalarDataLoader):
    def import_data(self, bounds):
        """
        Reads in data from an ERA5 NetCDF file.
        Renames coordinates to 'lat' and 'long'

        Args:
            bounds (Boundary): Initial boundary to limit the dataset to

        Returns:
            xr.Dataset:
                ERA5 wave dataset within limits of bounds.
                Dataset has coordinates 'lat', 'long', and variable 'swh'
        """
        # Open Dataset
        if len(self.files) == 1:    data = xr.open_dataset(self.files[0])
        else:                       data = xr.open_mfdataset(self.files)
        # Change column names
        data = data.rename({'longitude': 'lat',
                            'latitude': 'long'})
        # Limit to just swh data
        data = data['swh'].to_dataset()
        # Reverse order of lat as array goes from max to min
        data = data.reindex(lat=data.lat[::-1])
        # Trim to initial datapoints
        data = self.trim_datapoints(bounds, data=data)
        
        return data
