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
    @staticmethod
    def get_dataloader(name, bounds, params, min_dp=5):
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
        # Translate 'file' or 'folder' into 'files' key
        params = DataLoaderFactory.translate_file_input(params)
        
        # Add loader name to params
        params['dataloader_name'] = name
        params['min_dp'] = min_dp
        
        dataloader_requirements = {
            # Scalar
            'scalar_csv':   (ScalarCSVDataLoader, ['files']),
            'scalar_grf':   (ScalarGRFDataLoader, ['binary']),
            'binary_grf':   (ScalarGRFDataLoader,['binary']),
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
            'circle':       (ShapeDataLoader, []),
            'square':       (ShapeDataLoader, []),
            'gradient':     (ShapeDataLoader, []),
            'checkerboard': (ShapeDataLoader, []),
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
    
    @staticmethod
    def translate_file_input(params):
        '''
        Allows flexible file specification in params. Translates 'file' or 
        'folder' into 'files'
        
        Args:
            params (dict): 
                Dictionary of parameters written in config
        '''
        if 'file' in params:
            params['files'] = [params['file']]
            del params['file']
        elif 'folder' in params:
            folder = os.path.join(params['folder'], '') # Adds trailing slash if non-existant
            params['files'] = sorted(glob(folder+'*'))
            del params['folder']
        return params