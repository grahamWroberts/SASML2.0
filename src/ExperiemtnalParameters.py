import numpy as np
import VirtualInstrument.py

class ExperimentalParameters:

    def __init__():
        self.param_generators = {}
        self.calcs = {}
        self.morphologies = []
        self.instrument = None
        self.data = None

    def add_morphology(morph_name, generator):
        assert self.instrument is not None, "Error: Instrument not initialized. Please construct Virtual Instrument prior to adding morphologies"
        assert len(self.instrument) > 0, "Error: instrument has no reference curves, please either call VirtualIntrument.add_references(filenames) with a list of files or VirtualIntrument.add_configuration(q,I,dI,dq) to add a configuration explicitly."
        self.morphologies += [morph_name] 
        self.param_generators[morph_name] = generator
        self.calcs[morph_name] = self.instrument.construct_calculators[morph_name]

    def initialize(reference_files=None):
        self.instrument=VirtualInstrument.VirtualInstrument
        if reference_files is not None:
            self.instrument.add_references(reference_files)
        
