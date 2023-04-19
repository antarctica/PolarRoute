from polar_route.dataloaders.scalar.amsr import AMSRDataLoader
from polar_route.dataloaders.scalar.baltic_sea_ice import BalticSeaIceDataLoader
from polar_route.dataloaders.scalar.bsose_depth import BSOSEDepthDataLoader
from polar_route.dataloaders.scalar.bsose_sea_ice import BSOSESeaIceDataLoader
from polar_route.dataloaders.scalar.baltic_sea_ice import BalticSeaIceDataLoader
from polar_route.dataloaders.scalar.gebco import GEBCODataLoader
from polar_route.dataloaders.scalar.icenet import IceNetDataLoader
from polar_route.dataloaders.scalar.modis import MODISDataLoader
from polar_route.dataloaders.scalar.scalar_csv import ScalarCSVDataLoader
from polar_route.dataloaders.scalar.scalar_grf import ScalarGRFDataLoader
from polar_route.dataloaders.scalar.shape import ShapeDataLoader

from polar_route.dataloaders.vector.baltic_current import BalticCurrentDataLoader
from polar_route.dataloaders.vector.era5_wind import ERA5WindDataLoader
from polar_route.dataloaders.vector.north_sea_current import NorthSeaCurrentDataLoader
from polar_route.dataloaders.vector.oras5_current import ORAS5CurrentDataLoader
from polar_route.dataloaders.vector.sose import SOSEDataLoader
from polar_route.dataloaders.vector.vector_csv import VectorCSVDataLoader
from polar_route.dataloaders.vector.vector_grf import VectorGRFDataLoader

from polar_route.dataloaders.scalar.density import DensityDataLoader
from polar_route.dataloaders.scalar.thickness import ThicknessDataLoader

from glob import glob
import os


class DataLoaderFactory:
    '''
    Produces initialised DataLoader objects that can be used by the mesh to 
    quickly retrieve values within a boundary.
    '''    
    def get_dataloader(self, name, bounds, params, min_dp=5):
        '''
        Creates appropriate dataloader object based on name
        
        Args:
            name (str): 
                Name of data source/type. Must be one of following - 
                'scalar_csv', 'scalar_grf', 'binary_grf', 'amsr', 'bsose_sic',
                'bsose_depth', 'baltic_sic', 'gebco', 'icenet', 'modis', 
                'thickness', 'density', 'circle', 'square', 'gradient',
                'checkerboard', 'vector_csv', 'vector_grf', 'baltic_currents',
                'era5_wind', 'northsea_currents', 'oras5_currents', 'sose'
            bounds (Boundary): 
                Boundary object with initial mesh space&time limits
            params (dict): 
                Dictionary of parameters required by each dataloader
            min_dp (int):  
                Minimum datapoints required to get homogeneity condition

        Returns:
            (Scalar/Vector/LUT DataLoader): 
                DataLoader object of correct type, with required params set 
        '''
        # Cast name to lowercase to make case insensitive
        name = name.lower()
        # Add default values if they don't exist
        params = self.set_default_params(name, params, min_dp)
        
        dataloader_requirements = {
            # Scalar
            'scalar_csv':   (ScalarCSVDataLoader, ['files']),
            'scalar_grf':   (ScalarGRFDataLoader, []),
            'binary_grf':   (ScalarGRFDataLoader,[]),
            'amsr':         (AMSRDataLoader, ['files', 'hemisphere']),
            'bsose_sic':    (BSOSESeaIceDataLoader, ['files']),
            'bsose_depth':  (BSOSEDepthDataLoader, ['files']),
            'baltic_sic':   (BalticSeaIceDataLoader, ['files']),
            'gebco':        (GEBCODataLoader, ['files']),
            'icenet':       (IceNetDataLoader, ['files']),
            'modis':        (MODISDataLoader, ['files']),
            'thickness':    (ThicknessDataLoader, []),
            'density':      (DensityDataLoader, []),
            # Scalar - Abstract shapes
            'circle':       (ShapeDataLoader, ['shape', 'nx', 'ny', 'radius', 'centre']),
            'square':       (ShapeDataLoader, ['shape', 'nx', 'ny', 'side_length', 'centre']),
            'gradient':     (ShapeDataLoader, ['shape', 'nx', 'ny', 'vertical']),
            'checkerboard': (ShapeDataLoader, ['shape', 'nx', 'ny', 'gridsize']),
            # Vector
            'vector_csv':       (VectorCSVDataLoader, ['files']),
            'vector_grf':       (VectorGRFDataLoader, []),
            'baltic_currents':  (BalticCurrentDataLoader, ['files']),
            'era5_wind':        (ERA5WindDataLoader, ['files']),
            'northsea_currents':(NorthSeaCurrentDataLoader, ['files']),
            # TODO make it run from 'files'
            'oras5_currents':   (ORAS5CurrentDataLoader, ['files']),
            'sose':             (SOSEDataLoader, ['files'])

        }
        # If name is recognised as a dataloader
        if name in dataloader_requirements:
            # Set data loader and params required for it to work
            data_loader = dataloader_requirements[name][0]
            required_params = dataloader_requirements[name][1]
        else: 
            raise ValueError(f'{name} not in known list of DataLoaders')

        # Assert dataloader will get all required params to work
        assert all(key in params for key in required_params), \
            f'Dataloader {name} is missing some parameters! Requires {required_params}. Has {list(params.keys())}'

        # Create instance of dataloader
        return data_loader(bounds, params)
    
    def set_default_params(self, name, params, min_dp):
        '''
        Set default values for all dataloaders. 
        
        Args:
            name (str):
                Name of dataloader entry in dataloader_requirements. Used to
                specify default parameters for a specific dataloader.
            params (dict): 
                Dictionary containing attributes that are required for each 
                dataloader. 
            min_dp (int):
                Minimum number of datapoints required to return a homogeneity 
                condition. Passed in here so it can be added to params
            
        Returns:
            (dict): 
                Dictionary of attributes the dataloader will require, 
                completed with default values if not provided in config.
        '''
        # Save dataloader name in params
        params['dataloader_name'] = name
        
        if 'downsample_factors' not in params:
            params['downsample_factors'] = [1,1]

        if 'data_name' not in params:
            params['data_name'] = None

        if 'aggregate_type' not in params: 
            params['aggregate_type']  = 'MEAN'
            
        if 'min_dp' not in params:
            params['min_dp'] = min_dp
            
        if 'in_proj' not in params:
            params['in_proj'] = 'EPSG:4326'
            
        if 'out_proj' not in params:
            params['out_proj'] = 'EPSG:4326'
            
        if 'x_col' not in params:
            params['x_col'] = 'lat'

        if 'y_col' not in params:
            params['y_col'] = 'long'
            
        if 'file' in params:
            params['files'] = [params['file']]
        elif 'folder' in params:
            folder = os.path.join(params['folder'], '') # Adds trailing slash if non-existant
            params['files'] = sorted(glob(folder+'*'))
            
        # Set defaults for abstract data generators
        if name in ['circle', 'checkerboard', 'gradient']:
            params = self.set_default_shape_params(name, params)
        
        # Set defaults for GRF generators
        if name in ['binary_grf', 'scalar_grf', 'vector_grf']:
            params = self.set_default_grf_params(name, params)
        
        return params
    
    def set_default_shape_params(self, name, params):
        '''
        Set default values for abstract shape dataloaders. This function is
        seperated out from set_default_params() simply to reduce cognitive
        complexity, but is otherwise in the same format.
        
        Args:
            name (str):
                Name of shape entry in dataloader_requirements. Used to
                specify default parameters for the shape dataloader.
            params (dict): 
                Dictionary containing attributes that are required for the
                shape being loaded.
            
        Returns:
            (dict): 
                Dictionary of attributes the dataloader will require, 
                completed with default values if not provided in config.
        '''
        # Number of datapoints to populate per axis
        if 'nx' not in params:
            params['nx'] = 101
        if 'ny' not in params:
            params['ny'] = 101
            
        # Shape of abstract dataset
        if 'shape' not in params:
            params['shape'] = name
            
        # Define default circle parameters
        if name == 'circle':
            if 'radius' not in params:
                params['radius'] = 1
            if 'centre' not in params:
                params['centre'] = (None, None)
        # Define default square parameters
        elif name == 'square':
            if 'side_length' not in params:
                params['side_length'] = 1
            if 'centre' not in params:
                params['centre'] = (None, None)
        # Define default gradient params
        elif name == 'gradient':
            if 'vertical' not in params:
                params['vertical'] = True
        # Define default checkerboard params
        elif name == 'checkerboard':
            if 'gridsize' not in params:
                params['gridsize'] = (1,1)   
        
        
        return params

    def set_default_grf_params(self, name, params):
        # Params that all GRF dataloaders need
        if 'data_name' not in params:
            params['data_name'] = 'data'
        if 'seed' not in params:
            params['seed'] = None
        if 'size' not in params:
            params['size'] = 512
        if 'alpha' not in params:
            params['alpha'] = 3
        # Specific GRF loaders
        # If making a mask (e.g. land)
        if name == 'binary_grf':
            params['binary'] = True
            if 'min' not in params:
                params['min'] = 0
            if 'max' not in params:
                params['max'] = 1
            # If threshold not set, make it average of min/max val
            if 'threshold' not in params:
                params['threshold'] = 0.5
        # If making a scalar field
        elif name == 'scalar_grf':
            params['binary'] = False
            if 'min' not in params:
                params['min'] = -10
            if 'max' not in params:
                params['max'] = 10
            # If threshold not set, make it min/max vals
            if 'threshold' not in params:
                params['threshold'] = [0,1]
            if 'multiplier' not in params:
                params['multiplier'] = 1
            if 'offset' not in params:
                params['offset'] = 0
        # If making a vector field
        elif name == 'vector_grf':
            if 'min' not in params:
                params['min'] = 0
            if 'max' not in params:
                params['max'] = 10
            if 'vec_x' not in params:
                params['vec_x'] = 'uC'
            if 'vec_y' not in params:
                params['vec_y'] = 'vC'
                
        return params