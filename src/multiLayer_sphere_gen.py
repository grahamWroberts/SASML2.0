import numpy as np
import ParamGenerator as SG
import sasmodels.data
import sasmodels.core
import sasmodels.direct_model
import VirtualInstrument
from matplotlib import pyplot as plt


def random_MLS(count):
    n_shells = np.random.randint(5, 17, count)
    radii = np.random.uniform(20, 200, count)
    shell_thick = np.random.uniform(50,100,count)
    water_thick = np.random.uniform(10,50,count)
    sld= np.random.uniform(0.1, 10.0, count)
    return_list = [{"n_shells": n_shells[i],
                    "radius": radii[i],
                    "thick_shell": shell_thick[i],
                    "thick_solvent": water_thick[i],
                    "sld": sld[i]} for i in range(count)]
    return(return_list)

def map_MLS(indict):
    params_to_map = ["n_shells", "radius", "thick_shell", "thick_solvent", "sld"]
    default_dict = {"scale":1, "background": 0.001, "sld_solvent": 0.4, "radius_pd":.3, "radius_pd_n":40, "radius_pd_type":"schulz"}
    outdict = {key:indict[key] for key in params_to_map} | default_dict
    return(outdict)


if __name__ == "__main__":
   sg = SG.ParamGenerator(random_MLS, map_MLS)
   t_list, n_list = sg.sample(1000)
   print(n_list)
   model_name = 'multilayer_vesicle'
   reference_files = ["../../libal_sas/reference_sans/%s"%(s) for s in ["low_q.ABS", "med_q.ABS", "high_q.ABS"]]
   vi = VirtualInstrument.VirtualInstrument()
   vi.add_references(reference_files)
   calcs, sds = vi.construct_calculators(model_name)
   curves = [vi.generate(model_name, kw, calcs, sds)[0] for kw in n_list]
   #print(curves[0])





###   plt.xscale('log')
###   plt.yscale('log')
###   plt.xlabel('$q \,(\mathrm{\AA})^{-1}$')
###   plt.ylabel('$I(q)\, (\mathrm{cm})^{-1}$')
###   plt.title("random multilayer vesicles")
###   ind = 0
###   for c in curves:
###       inds = c.index
###       plt.scatter(c.index, c[:], label = n_list[ind]["n_shells"])
###       ind += 1
###   plt.legend()
###   plt.savefig("example_mlv.png")
   #plt.show()

   

