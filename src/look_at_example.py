import numpy as np
import ParamGenerator as SG
import sasmodels.data
import sasmodels.core
import sasmodels.direct_model
import VirtualInstrument
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

targets = ["../../libal_sas/libal/reference_sans/%s_q_trimmed.ABS"%(t) for t in ['high']]

#targets = ["../../libal_sas/libal/reference_sans/%s_q_trimmed.ABS"%(t) for t in ["low", "med", 'high']]

model_name = 'cylinder'
kw = {'radius': 100,
      'length': 600}
for t in targets:
   vi = VirtualInstrument.VirtualInstrument()
   vi.add_references([t])
   calcs, sds = vi.construct_calculators(model_name)
   _, curve, I = vi.generate(model_name, kw, calcs, sds)
   print(curve)
   plt.clf()
   plt.xscale('log')
   plt.yscale('log')
   q = curve.index
   I = curve[q]
   plt.scatter(q,I)
   plt.show()
