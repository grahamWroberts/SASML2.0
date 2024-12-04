import numpy as np

class Sampler:

    def __init__(self, name=None, user_parameters=None, structural_parameters=None, parameterizer=None):
        self.name = name 
        self.user_parameters = user_parameters
        self.structural_parameters = structural_parameters
        self.parameterizer = parameterizer
    
    def random_params(self, count=1):
        rdict = {}
        assert (self.user_parameters is not None),"Error internal parameters not initialized"
        for key in self.user_parameters.keys():
            assert ("minimum" in self.user_parameters[key] and "maximum" in self.user_parameters[key]), "key %s does not have minmum and maximum range"%(key)
            rdict[key] = np.random.uniform(self.user_parameters[key]['minimum'], self.user_parameters[key]['maximum'], count)
        sample = [{k:rdict[k][i] for k in rdict.keys()} for i in range(count)]
        return(sample)
    
    def map_structural_parameters(self, sample):
        assert self.parameterizer is not None, "Structural parameters not defined"
        assert self.parameterizer is not None, "No function mapping user parameters to structural parameters"
        assert np.all([key in self.structural_parameters for k in self.parameterizer]), "parameterizer mapping to unrecognized parameter"
        new_params = self.structural_parameters.copy()
        for key in self.parameterizer:
            new_params[key] = parameterizer[key](sample)
        return(new_params)

    def random_structures(self, count=1):
        

    
