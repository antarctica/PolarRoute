import sobol
import numpy

class Sampler:

    def __init__(self ):
        self.samples= []
        self.mapped_samples= []


    def generate_samples ( self, d , n):
        self.samples = sobol.sample(dimension=d, n_points=n)
        # print (self.samples)

    def map_samples_to_range (self, ranges):
        
        for sample in self.samples:
            for i in range (len(ranges)):
                self.mapped_samples. append (ranges [i] [0] + sample[i]* (ranges[i] [1] - ranges [i][0]))
        self.mapped_samples = numpy.array(self.mapped_samples).reshape ( len (self.samples), len(ranges))
        
       

if __name__=='__main__':
    sampler = Sampler()
    sampler.generate_samples(2,3)
    ranges = [[10,20] , [100,200]]
    sampler.map_samples_to_range(ranges)
    print (sampler.mapped_samples)