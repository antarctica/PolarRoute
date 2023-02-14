from abc import ABCMeta, abstractmethod
from polar_route.Boundary import Boundary

class DataLoaderInterface(metaclass=ABCMeta):
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__init__')           and 
                callable(subclass.__init__)             and
                hasattr(subclass, 'import_data')        and
                callable(subclass.import_data)          and
                hasattr(subclass, 'get_datapoints')     and
                callable(subclass.get_datapoints)       and
                hasattr(subclass, 'get_value')          and
                callable(subclass.get_value)            and
                hasattr(subclass, 'get_hom_condition')  and
                callable(subclass.get_hom_condition)    and
                hasattr(subclass, 'reproject')          and
                callable(subclass.reproject)            and
                hasattr(subclass, 'downsample')         and
                callable(subclass.downsample)           and
                hasattr(subclass, 'get_data_col_name')  and
                callable(subclass.get_data_col_name)    and
                hasattr(subclass, 'set_data_col_name')  and
                callable(subclass.set_data_col_name)    or
                NotImplemented)
        
    @abstractmethod    
    def __init__(self):
        ''' Set up main processes to read and store data '''
        raise NotImplementedError

    @abstractmethod  
    def import_data(self):
        ''' Import raw data from file and 
        return xr.Dataset or pd.DataFrame '''
        raise NotImplementedError
    
    @abstractmethod  
    def get_datapoints(self, bounds: Boundary):
        ''' Retrieve datapoints within boundary and
        return pd.Series'''
        raise NotImplementedError
    
    @abstractmethod  
    def get_value(self, bounds: Boundary, agg_type: str):
        ''' Reads values within boundary and 
        returns aggregated value as np.float64'''
        raise NotImplementedError

    @abstractmethod  
    def get_hom_condition(self, bounds: Boundary, 
                          splitting_conds: dict, agg_type: str):
        ''' Reads values within boundary to determine if data is
        homogeneous or heterogeneous. Return as str'''
        raise NotImplementedError
    
    @abstractmethod  
    def reproject(self, in_proj: str, out_proj: str, 
                  x_col: str, y_col: str):
        ''' Reprojects raw data into a common projection and 
        return as pd.DataFrame'''
        raise NotImplementedError
    
    @abstractmethod  
    def downsample(self, agg_type: str):
        ''' Downsamples data for easier manipulation of dense datasets and
        return xr.Dataset or pd.DataFrame'''
        raise NotImplementedError
    
    @abstractmethod  
    def get_data_col_name(self):
        ''' Get name of data column(s) and 
        return as str '''
        raise NotImplementedError
    
    @abstractmethod  
    def set_data_col_name(self, new_name: str):
        ''' Changes name of data column(s) and 
        return xr.Dataset or pd.DataFrame '''
        raise NotImplementedError