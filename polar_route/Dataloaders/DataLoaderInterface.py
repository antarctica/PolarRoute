from abc import ABCMeta, abstractmethod
from polar_route.Boundary import Boundary

class DataLoaderInterface(metaclass=ABCMeta):
    '''
    The interface through which the mesh operates on the dataloaders.
    Dataloaders store all data internally, and will retrieve
    aggregated values and homogeneity conditions when called.
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_value')          and
                callable(subclass.get_value)            and
                hasattr(subclass, 'get_hom_condition')  and
                callable(subclass.get_hom_condition)   or
                NotImplemented)
    
    @abstractmethod  
    def get_value(self, bounds: Boundary, agg_type: str):
        ''' Reads values within boundary and 
        returns aggregated value as np.float64'''
        raise NotImplementedError

    @abstractmethod  
    def get_hom_condition(self, bounds: Boundary, splitting_conds: dict):
        ''' Reads values within boundary to determine if data is
        homogeneous or heterogeneous. Return as str'''
        raise NotImplementedError