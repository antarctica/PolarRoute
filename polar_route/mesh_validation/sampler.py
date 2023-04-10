import math
from scipy.stats import qmc
import numpy as np

class Sampler:

    def __init__(self, d , n ):
      
        self.dimensions = d
        self.number_of_samples = n


    def generate_samples ( self, ranges):
        sampler =  qmc.Sobol(d=self.dimensions)
        n_points_log_2= int (math.log2(self.number_of_samples))
        samples = sampler.random_base2(m=n_points_log_2)
        mapped_samples = []
        # map samples to ranges
        for sample in samples:
            for i in range (len(ranges)):
                mapped_samples. append (ranges [i] [0] + sample[i]* (ranges[i] [1] - ranges [i][0]))
        return np.array(mapped_samples).reshape ( len (samples), len(ranges))
        
        
       
