import numpy as np
import pandas as pd
import xarray as xr

import sasmodels.data
import sasmodels.core
import sasmodels.direct_model

def extract_Iq(fn):
    try:
       df = pd.read_csv(fn)
    except:
       df = pd.read_csv(fn, delim_whitespace=True)
    print(df.columns)
    q = list(df.q)
    I = list(df.I)
    dq = list(df.dq)
    dI = list(df.dI)
    return(q, I, dq, dI)

class VirtualInstrument:
    def __init__(self, noise=1e-9):
        self.reference_data = []
        self.noise = noise

    def add_references(self,filenames):
        if type(filenames) == list:
            for fn in filenames:
               q, I, dq, dI = extract_Iq(fn)
               self.add_configuration(q,I,dI,dq, reset=False)

        else:
            q, I, dq, dI = extract_Iq(filenames)
            self.add_configuration(q,I,dI,dq, rest=False)


    def add_configuration(self,q,I,dI,dq,reset=True):
        '''Read in reference data for an instrument configuration'''
        if reset:
            self.reference_data = []
        data = sasmodels.data.Data1D(
            x=np.array(q),
            y=np.array(I),
            dy=np.array(dI),
            dx=np.array(dq),
        )
        self.reference_data.append(data)

    def generate(self, label, kw, calculators, sasdatas):
        I_noiseless_list = []
        I_list = []
        dI_list = []
        for sasdata,calc in zip(sasdatas,calculators):
            I_noiseless = calc(**kw)
            
            I_noiseless[np.where(np.less(I_noiseless, 0.001))[0]] = 0.001
            dI_model = sasdata.dy*np.sqrt(I_noiseless/sasdata.y)
            mean_var= np.mean(dI_model*dI_model/I_noiseless)
            # dI = sasdata.dy*np.sqrt(noise*noise/mean_var)
            dI = sasdata.dy*self.noise/mean_var
            
            I = np.random.normal(loc=I_noiseless,scale=dI)
            I[np.where(np.less(I, 0.001))[0]] = 0.001
            
            I_noiseless = pd.Series(data=I_noiseless,index=sasdata.x)
            I = pd.Series(data=I,index=sasdata.x)
            dI = pd.Series(data=dI,index=sasdata.x)
            
            I_list.append(I)
            I_noiseless_list.append(I_noiseless)
            dI_list.append(dI)
            
        I           = pd.concat(I_list).sort_index()
        I_noiseless = pd.concat(I_noiseless_list).sort_index()
        dI          = pd.concat(dI_list).sort_index()
        return I,I_noiseless,dI

    def construct_calculators(self, model_name):
        calculators = []
        sasdatas = []
        for sasdata in self.reference_data:
            model_info    = sasmodels.core.load_model_info(model_name)
            print(sasdata)
            kernel        = sasmodels.core.build_model(model_info)
            calculator    = sasmodels.direct_model.DirectModel(sasdata,kernel)
            calculators.append(calculator)
            sasdatas.append(sasdata)
        return(calculators, sasdatas)
