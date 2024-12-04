import numpy as np

class ParamGenerator:

    def __init__(self, user_parameter_function=None, parameterizer=None):
        self.UPF = user_parameter_function #a function that takes an integer, N, and returns a list of N dictionaries
        self.SPF = parameterizer #a function that takes a dictionary of suer_specified_parameters and returns a list of structural parameters for SASView

    def sample(self, count=1):
        user_samp = self.UPF(count)
        struct_samp = [self.SPF(up) for up in user_samp]
        return(user_samp, struct_samp)
