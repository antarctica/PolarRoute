import math
from scipy.stats import qmc
import numpy as np

class Sampler:
    """
    a class that generates samples based on Sobol sequences, which fills the space in a highly uniform way and guarantees a better coverage of the sampling space. 
    Attributes:
        dimensions (int): an integer representing the dimensions of each sample 
        number_of_samples (int): an integer representing the number of the generated samples


    """
    def __init__(self, d , n ):
      
        self.dimensions = d
        self.number_of_samples = n


    def generate_samples ( self, ranges):
        """

          generates samples within the provided ranges array, the length of the ranges should equal to self.dimensions

            Args:
              ranges (float[]): an array that contains the range that each sample dimension should fall within
            Returns:
                distance (float): the RMSE between the actaul data value and the mesh's agg_value.

        """
        if len(ranges) != self.dimensions:
            raise ValueError("ranges length should be equal to the sampler dimension") 

        sampler =  qmc.Sobol(d=self.dimensions)
        samples = sampler.random(n=self.number_of_samples)
        mapped_samples = []
        # map samples to ranges
        for sample in samples:
            for i in range (len(ranges)):
                mapped_samples. append (ranges [i] [0] + sample[i]* (ranges[i] [1] - ranges [i][0]))
        return np.array(mapped_samples).reshape ( len (samples), len(ranges))
        
        
       
