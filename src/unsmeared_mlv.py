import numpy as np
import ParamGenerator as SG
import sasmodels.data
import sasmodels.core
import sasmodels.direct_model
import VirtualInstrument
from matplotlib import pyplot as plt
import pandas as pd

model_name = 'multilayer_vesicle'
model_info = sasmodels.core.load_model_info(model_name)
kernel = sasmodels.core.build_model(model_info)
reference_files = ["../../libal_sas/reference_sans/%s"%(s) for s in ["low_q.ABS", "med_q.ABS", "high_q.ABS"]]
keys = ['low-q', 'medium-q', 'high-q']
vis = []
curves = []
noiseless_curves = []
specific_dict = {'radius': 100,
                 'n_shells': 20,
                 'thick_shell': 50,
                 'thick_solvent': 10,
                 'sld_solvent': 6.4,
                 'sld': 1.0,
                 'scale': 1,
                 'background': 0.001,
                 'volfraction':0.05,
                 'radius_pd': 0.25,
                 'radius_pd_n': 40.5,
                 'radius_pd_nsigma': 4.5,
                 'radius_pd_type': 'schulz'}
allcalcs = []
allsds = []
allqs = None
concurve = None
for i in range(len(reference_files)):
   indf = pd.read_csv(reference_files[i], delim_whitespace=True)
   q = list(indf.q)
   d1d = sasmodels.data.Data1D(x=np.array(q))
   temp_calc = sasmodels.direct_model.DirectModel(d1d, kernel)
   allsds += [temp_calc]
   allcalcs += [temp_calc]
   vi = VirtualInstrument.VirtualInstrument()
   vis += [vi]
   vi.add_references([reference_files[i]])
   calcs, sds = vi.construct_calculators(model_name)
   cs = vi.generate(model_name, specific_dict, calcs, sds)
   temp_curve = temp_calc(**specific_dict)
   allqs = np.append(allqs, q) if allqs is not None else q
   concurve = np.append(concurve, temp_curve) if concurve is not None else temp_curve
   print(temp_curve)
   curves += [cs[0]]
   noiseless_curves += [cs[1]]
   outmap = np.zeros((cs[0].shape[0],2))
   outmap[:,0] = cs[0].index
   outmap[:,1] = cs[1][:]
   plt.xscale('log')
   plt.yscale('log')
   plt.xlabel('$q \,(\mathrm{\AA})^{-1}$')
   plt.ylabel('$I(q)\, (\mathrm{cm})^{-1}$')
   plt.title("multilayer vesicle %s"%(keys[i]))
   plt.scatter(cs[0].index, cs[0][:], label = 'noisy', s=2)
   plt.scatter(cs[0].index, cs[1][:], label = 'noiseless', s=2)
   plt.scatter(cs[0].index, temp_curve, label = 'raw', s=2)
   plt.legend()
   plt.savefig('%s-ex-multilayer-vesicle.pdf'%(keys[i]))
   plt.clf()
   np.savetxt('%s-multilayer-vesicle.csv'%(keys[i]), outmap)
#newcs = vi.generate(model_name, specific_dict, allcalcs, allsds)
#print(newcs)
inds = np.argsort(allqs)
allqs = allqs[inds]
concurve = concurve[inds]
plt.clf()
plt.xscale('log')
plt.yscale('log')
print(allqs)
print(concurve)
plt.plot(allqs, concurve)
plt.xlabel('$q \,(\mathrm{\AA})^{-1}$')
plt.ylabel('$I(q)\, (\mathrm{cm})^{-1}$')
plt.title("multilayer vesicle with no smearing")
plt.savefig("noiseless_curve.pdf")

   
