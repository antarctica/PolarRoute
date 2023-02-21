from polar_route.Dataloaders.Scalar.AMSR import AMSRDataLoader
from polar_route.Dataloaders.Scalar.BalticSeaIce import BalticSeaIceDataLoader
from polar_route.Dataloaders.Scalar.BSOSEDepth import BSOSEDepthDataLoader
from polar_route.Dataloaders.Scalar.BSOSESeaIce import BSOSESeaIceDataLoader
from polar_route.Dataloaders.Scalar.BalticSeaIce import BalticSeaIceDataLoader
from polar_route.Dataloaders.Scalar.GEBCO import GEBCODataLoader
from polar_route.Dataloaders.Scalar.IceNet import IceNetDataLoader
from polar_route.Dataloaders.Scalar.MODIS import MODISDataLoader
from polar_route.Dataloaders.Scalar.ScalarCSV import ScalarCSVDataLoader
from polar_route.Dataloaders.Scalar.Shape import ShapeDataLoader

from polar_route.Dataloaders.Vector.BalticCurrent import BalticCurrentDataLoader
from polar_route.Dataloaders.Vector.ERA5Wind import ERA5WindDataLoader
from polar_route.Dataloaders.Vector.NorthSeaCurrent import NorthSeaCurrentDataLoader
from polar_route.Dataloaders.Vector.ORAS5Current import ORAS5CurrentDataLoader
from polar_route.Dataloaders.Vector.SOSE import SOSEDataLoader
from polar_route.Dataloaders.Vector.VectorCSV import VectorCSVDataLoader

from polar_route.Dataloaders.Scalar.Density import DensityDataLoader
from polar_route.Dataloaders.Scalar.Thickness import ThicknessDataLoader
# from .LookupTable.Thickness import ThicknessDataLoader
# from .LookupTable.Density import DensityDataLoader

from polar_route.Boundary import Boundary



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
            'scalarcsv':(ScalarCSVDataLoader, ['file']),
            'amsr':        (AMSRDataLoader, ['file', 'hemisphere']),
            'amsr_folder': (AMSRDataLoader, ['folder', 'hemisphere']),
            'bsose_sic':   (BSOSESeaIceDataLoader, ['file']),
            'bsose_depth': (BSOSEDepthDataLoader, ['file']),
            'baltic_sic':  (BalticSeaIceDataLoader, ['file']),
            'gebco':       (GEBCODataLoader, ['file']),
            'icenet':      (IceNetDataLoader, ['file']),
            'modis':       (MODISDataLoader, ['file']),
            # Scalar - Abstract shapes
            'circle':       (ShapeDataLoader, ['shape', 'nx', 'ny', 'radius', 'centre']),
            'square':       (ShapeDataLoader, ['shape', 'nx', 'ny', 'side_length', 'centre']),
            'gradient':     (ShapeDataLoader, ['shape', 'nx', 'ny', 'vertical']),
            'checkerboard': (ShapeDataLoader, ['shape', 'nx', 'ny', 'gridsize']),
            # Vector
            'vectorcsv':     (VectorCSVDataLoader, ['file']),
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
    
if __name__=='__main__':

    long_range = [-70, -50]
    lat_range = [-65, -60]
    
    bounds = Boundary(lat_range, long_range, ['2019-01-01','2019-01-14'])
    # bad_lat_range, bad_long_range = polygon_str_to_boundaries(
    #     # "POLYGON ((-70 -61.9921875, -70 -61.953125, -69.921875 -61.953125,  -69.921875 -61.9921875, -70 -61.9921875))"                    # 1316
    #     # "POLYGON ((-69.921875 -61.9921875, -69.921875 -61.953125, -69.84375 -61.953125, -69.84375 -61.9921875, -69.921875 -61.9921875))"  # 1317
    #     # "POLYGON ((-70 -62.03125, -70 -61.9921875, -69.921875 -61.9921875, -69.921875 -62.03125, -70 -62.03125))"                         # 1318
    #     # "POLYGON ((-69.921875 -62.03125, -69.921875 -61.9921875, -69.84375 -61.9921875, -69.84375 -62.03125, -69.921875 -62.03125))"      # 1319
    #     "POLYGON ((-70 -62.03125, -70 -61.953125, -69.84375 -61.953125, -69.84375 -62.03125, -70 -62.03125))"          # Total
    #     # "POLYGON ((-59.375 -60.703125, -59.375 -60.625, -59.21875 -60.625, -59.21875 -60.703125, -59.375 -60.703125))"          # Total
    #     "POLYGON ((-61.5625 -63.828125, -61.5625 -63.75, -61.40625 -63.75, -61.40625 -63.828125, -61.5625 -63.828125))"
    # )
    # bad_cb_bounds = Boundary(bad_lat_range, bad_long_range, ['2013-03-01','2013-03-14'])

    
    # ............... SCALAR DATA LOADERS ............... #
    
    if False: # Run GEBCO
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/bathymetry/GEBCO/gebco_2022_n-40.0_s-90.0_w-140.0_e0.0.nc',
            'downsample_factors': (5,5),
            'data_name': 'elevation',
            'aggregate_type': 'MAX'
        }
        split_conds = {
            'threshold': 620,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        gebco = factory.get_dataloader('GEBCO', bounds, params, min_dp = 5)
        print(gebco.get_value(bounds))
        print(gebco.get_hom_condition(bounds, split_conds))
    if False: # Run AMSR
        params = {
            'folder': '/home/habbot/Documents/Work/PolarRoute/datastore/sic/amsr_south/',
            # 'file': 'PolarRoute/datastore/sic/amsr_south/asi-AMSR2-s6250-20201110-v5.4.nc',
            'data_name': 'SIC',
            'aggregate_type': 'MEAN',
            'hemisphere': 'South'
        }
        split_conds = {
            'threshold': 35,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        amsr = factory.get_dataloader('AMSR_folder', bounds, params, min_dp = 5)
        print(amsr.get_value(bounds))
        print(amsr.get_hom_condition(bounds, split_conds))
    if False: # Run BSOSE Sea Ice
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/sic/bsose/bsose_i122_2013to2017_1day_SeaIceArea.nc',
            'data_name': 'SIC',
            'aggregate_type': 'MEAN'
        }
        split_conds = {
            'threshold': 0.35,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        bsose_sic = factory.get_dataloader('bsose_sic', bounds, params, min_dp = 5)
        print(bsose_sic.get_value(bounds))
        print(bsose_sic.get_hom_condition(bounds, split_conds))
    if False: # Run BSOSE Depth         - NEED DATA TO TEST
        pass
    if False: # Run Baltic Sea Ice
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/sic/baltic/BalticIceMar05.nc',
            'data_name': 'SIC',
            'aggregate_type': 'MEAN'
        }
        split_conds = {
            'threshold': 0.35,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        baltic_sic = factory.get_dataloader('baltic_sic', bounds, params, min_dp = 5)
        print(baltic_sic.get_value(bounds))
        print(baltic_sic.get_hom_condition(bounds, split_conds))
    if False: # Run MODIS               - NEED DATA TO TEST
        pass
    if False: # Run Dummy Scalar        - NEED DATA TO TEST
        pass
    # ............... VECTOR DATA LOADERS ............... #
    if False: # Run SOSE
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/currents/sose_currents/SOSE_surface_velocity_6yearMean_2005-2010.nc',
            'aggregate_type': 'MEAN'
        }
        sose = factory.get_dataloader('SOSE', bounds, params, min_dp = 5)
        print(sose.get_value(bounds))
    if False: # Run Baltic Currents     - NEED DATA TO TEST
        pass
    if False: # Run ERA5 Wind
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/wind/era5_wind_2013.nc',
            'aggregate_type': 'MEAN'
        }
        era5 = factory.get_dataloader('era5_wind', bounds, params, min_dp = 5)
        print(era5.get_value(bounds))
    if False: # Run North Sea Currents
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/currents/north_atlantic/CS3_POLCOMS2006_11.nc',
            'aggregate_type': 'MEAN'
        }
        northsea_currents = factory.get_dataloader('northsea_currents', bounds, params, min_dp = 5)
        print(northsea_currents.get_value(bounds))
    if False: # Run ORAS5 Currents
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/currents/oras5/oras5_2019.nc',
            'aggregate_type': 'MEAN'
        }
        oras5 = factory.get_dataloader('oras5_currents', bounds, params, min_dp = 5)
        print(oras5.get_value(bounds))
    if False: # Run Dummy Vector
        pass
    # ............... LOOKUP TABLE DATA LOADERS ............... #
    if True: # Run Thickness
        params = {
            'data_name': 'thickness',
        }
  
        split_conds = {
            'threshold': 0.5,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        
        thickness = DataLoaderFactory().get_dataloader('thickness', bounds, params, min_dp = 5)
        print(thickness.get_value(bounds))
        print(thickness.get_hom_condition(bounds, split_conds))
    if True: # Run Density
        params = {
            'data_name': 'density',
        }
  
        split_conds = {
            'threshold': 900,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        
        density = DataLoaderFactory().get_dataloader('density', bounds, params, min_dp = 5)
        print(density.get_value(bounds))
        print(density.get_hom_condition(bounds, split_conds))

    # ............... ABSTRACT SHAPE DATA LOADERS ............... #

    if False: # Run Circle
        params = {
            "data_name": "dummy_data",
            "value_fill_types": "parent",
            "nx": 201,
            "ny": 201,
            "radius": 3,
            "centre": [-65, -70],
            "radius": 2,
            "centre": [-62.5, -60],
        }

        split_conds = {
            'threshold': 0.5,
            'upper_bound': 0.8,
            'upper_bound': 0.99,
            'lower_bound': 0.01
        }

        circle = factory.get_dataloader('circle', bounds, params, min_dp = 5)
    if False: # Run Gradient
        params = {
            'n': 11,
            'vertical': False
        }
        split_conds = {
            'threshold': 0.5,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        gradient = factory.get_dataloader('gradient', bounds, params, min_dp = 1)
        
        print(gradient.get_value(bounds))
        print(gradient.get_hom_condition(bounds, split_conds))
    if False: # Run Checkerboard
        params = {
            'nx': 201,
            'ny': 201,
            'gridsize': (6,3)
        }
        split_conds = {
            'threshold': 0.5,
            'upper_bound': 0.85,
            'lower_bound': 0.15
        }
        checkerboard = factory.get_dataloader('checkerboard', bounds, params, min_dp = 5)
        
        print(checkerboard.get_value(bounds))
        print(checkerboard.get_hom_condition(bounds, split_conds))
            
    print('hi')