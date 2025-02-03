import numpy as np
import ParamGenerator as SG
import sasmodels.data
import sasmodels.core
import sasmodels.direct_model
import VirtualInstrument
from matplotlib import pyplot as plt


def random_MLS(count):
    n_shells = np.random.randint(5, 30, count)
    radii = np.random.uniform(50, 2000, count)
    shell_thick = np.random.uniform(25,100,count)
    water_thick = np.random.uniform(10,15,count)
    sld= np.random.uniform(0.5, 2.0, count)
    return_list = [{"n_shells": n_shells[i],
                    "radius": radii[i],
                    "thick_shell": shell_thick[i],
                    "thick_solvent": water_thick[i],
                    "sld": sld[i]} for i in range(count)]
    return(return_list)

def map_MLS(indict):
    params_to_map = ["n_shells", "radius", "thick_shell", "thick_solvent", "sld"]
    default_dict = {"scale":1, "background": 0.001, "sld_solvent": 6.4, "radius_pd":.3, "radius_pd_n":40, "radius_pd_type":"schulz"}
    outdict = {key:indict[key] for key in params_to_map} | default_dict
    return(outdict)


if __name__ == "__main__":
   sg = SG.ParamGenerator(random_MLS, map_MLS)
   t_list, n_list = sg.sample(5)
   print(n_list)
   model_name = 'multilayer_vesicle'
   reference_files = ["../../libal_sas/reference_sans/%s"%(s) for s in ["low_q.ABS", "med_q.ABS", "high_q.ABS"]]
   vi = VirtualInstrument.VirtualInstrument()
   vi.add_references(reference_files)
   calcs, sds = vi.construct_calculators(model_name)
   curves = [vi.generate(model_name, kw, calcs, sds)[0] for kw in n_list]
   print("C range %0.2e - %0.2e"%(np.min(curves[0].index), np.max(curves[0].index)))
   print("effective range %0.2e - %0.2e"%((2*np.pi)/np.max(curves[0].index), (2*np.pi)/np.min(curves[0].index)))
   #print(curves[0])





   plt.xscale('log')
   plt.yscale('log')
   plt.xlabel('$q \,(\mathrm{\AA})^{-1}$')
   plt.ylabel('$I(q)\, (\mathrm{cm})^{-1}$')
   plt.title("random multilayer vesicles")
   ind = 0
   for c in curves:
       inds = c.index
       plt.scatter(c.index, c[:], label = "%d %0.2f %0.2f %0.2f"%(n_list[ind]["n_shells"], n_list[ind]["radius"], n_list[ind]["thick_shell"], n_list[ind]["thick_solvent"]))
       ind += 1
   plt.legend()
   plt.savefig("example_mlv.png")
   #plt.show()

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
   #curves = [vi.generate(model_name, kw, calcs, sds)[0] for kw in n_list]
   curve, curve_noiseless, dI = vi.generate(model_name, specific_dict, calcs, sds)
   plt.clf()
   plt.xscale('log')
   plt.yscale('log')
   plt.xlabel('$q \,(\mathrm{\AA})^{-1}$')
   plt.ylabel('$I(q)\, (\mathrm{cm})^{-1}$')
   plt.title("multilayer vesicle")
   plt.scatter(curve.index, curve[:], label = 'noiseless', s=1, alpha=.1)
   plt.scatter(curve.index, curve[:], label = 'noise', marker = '^', s=1)
   plt.legend()
   plt.savefig("mlv.png")
   out_mlv = np.zeros((curve.shape[0], 2))
   out_mlv[:,0] = curve.index
   out_mlv[:,1] = curve[:]
   np.savetxt('ex_mlv.csv', out_mlv)

                    

   

