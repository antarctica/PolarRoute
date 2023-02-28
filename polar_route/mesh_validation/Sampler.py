import sobol
import numpy

class Sampler:

    def __init__(self, d , n ):
      
        self.dimensions = d
        self.number_of_samples = n


    def generate_samples ( self, ranges):
        samples = sobol.sample(dimension=self.dimensions, n_points=self.number_of_samples)
        mapped_samples = []
        # map samples to ranges
        for sample in samples:
            for i in range (len(ranges)):
                mapped_samples. append (ranges [i] [0] + sample[i]* (ranges[i] [1] - ranges [i][0]))
        return numpy.array(mapped_samples).reshape ( len (samples), len(ranges))
        
       

if __name__=='__main__':
    sampler = Sampler()
    sampler.generate_samples(2,3)
    ranges = [[10,20] , [100,200]]
    sampler.map_samples_to_range(ranges)
    print (sampler.mapped_samples)