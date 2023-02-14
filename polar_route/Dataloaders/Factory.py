from .Scalar.AMSR import AMSRDataLoader
from .Scalar.BalticSeaIce import BalticSeaIceDataLoader
from .Scalar.BSOSEDepth import BSOSEDepthDataLoader
from .Scalar.BSOSESeaIce import BSOSESeaIceDataLoader
from .Scalar.BalticSeaIce import BalticSeaIceDataLoader
from .Scalar.GEBCO import GEBCODataLoader
from .Scalar.MODIS import MODISDataLoader
from .Scalar.ScalarCSV import ScalarCSVDataLoader
from .Scalar.Shape import ShapeDataLoader

from .Vector.BalticCurrent import BalticCurrentDataLoader
from .Vector.ERA5Wind import ERA5WindDataLoader
from .Vector.NorthSeaCurrent import NorthSeaCurrentDataLoader
from .Vector.ORAS5Current import ORAS5CurrentDataLoader
from .Vector.SOSE import SOSEDataLoader
from .Vector.VectorCSV import VectorCSVDataLoader

from .Scalar.Density import DensityDataLoader
from .Scalar.Thickness import ThicknessDataLoader
# from .LookupTable.Thickness import ThicknessDataLoader
# from .LookupTable.Density import DensityDataLoader



class DataLoaderFactory:
    '''
    Produces initialised DataLoader objects
    '''    
    def get_dataloader(self, name, bounds, params, min_dp=5):
        '''
        Creates appropriate dataloader object based on name
        
        Args:
            name (str): 
                Name of data source/type. Must be one of following - 
                'GEBCO','AMSR','SOSE','thickness','density',
                'GRFScalar','GRFVector','GRFMask'
            bounds (Boundary): 
                Boundary object with initial mesh space&time limits
            params (dict): 
                Dictionary of parameters required by each dataloader
            min_dp (int):  
                Minimum datapoints required to get homogeneity condition

        Returns:
            data_loader (Scalar/Vector/LUT DataLoader): 
                DataLoader object of correct type, with required params set 
        '''
        # Cast name to lowercase to make case insensitive
        name = name.lower()
        # Add default values if they don't exist
        params = self.set_default_params(name, params, min_dp)
        
        dataloader_requirements = {
            # Scalar
            'dummy_scalar':(ScalarCSVDataLoader, ['file']),
            'amsr':        (AMSRDataLoader, ['file', 'hemisphere']),
            'amsr_folder': (AMSRDataLoader, ['folder', 'hemisphere']),
            'bsose_sic':   (BSOSESeaIceDataLoader, ['file']),
            'bsose_depth': (BSOSEDepthDataLoader, ['file']),
            'baltic_sic':  (BalticSeaIceDataLoader, ['file']),
            'gebco':       (GEBCODataLoader, ['file']),
            'modis':       (MODISDataLoader, ['file']),
            # Scalar - Abstract shapes
            'circle':       (ShapeDataLoader, ['shape', 'nx', 'ny', 'radius', 'centre']),
            'square':       (ShapeDataLoader, ['shape', 'nx', 'ny', 'side_length', 'centre']),
            'gradient':     (ShapeDataLoader, ['shape', 'nx', 'ny', 'vertical']),
            'checkerboard': (ShapeDataLoader, ['shape', 'nx', 'ny', 'gridsize']),
            # Vector
            'dummy_vector':     (VectorCSVDataLoader, ['file']),
            'baltic_currents':  (BalticCurrentDataLoader, ['file']),
            'era5_wind':        (ERA5WindDataLoader, ['file']),
            'northsea_currents':(NorthSeaCurrentDataLoader, ['file']),
            'oras5_currents':   (ORAS5CurrentDataLoader, ['file_u', 'file_v']),
            'sose':             (SOSEDataLoader, ['file']),
            # Lookup Table
            # TODO actually make these LUT
            'thickness': (ThicknessDataLoader, []),
            'density':   (DensityDataLoader, [])
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
        Set default values for all dataloaders
        '''
        
        if 'downsample_factors' not in params:
            params['downsample_factors'] = (1,1)

        if 'data_name' not in params:
            params['data_name'] = name

        if 'aggregate_type' not in params: 
            params['aggregate_type']  = 'MEAN'
            
        if 'min_dp' not in params:
            params['min_dp'] = min_dp
            
        # Set defaults for abstract data generators
        if name in ['circle', 'checkerboard', 'gradient']:
            params = self.set_default_shape_params(name, params)
                
        return params
    
    def set_default_shape_params(self, name, params):
        '''
        Set default values for abstract shape dataloaders
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
