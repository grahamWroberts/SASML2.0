import numpy as np
import ParamGenerator as SG
import sasmodels.data
import sasmodels.core
import sasmodels.direct_model
import VirtualInstrument


def random_MLS(count):
    n_shells = np.random.randint(2, 9, count)
    radii = np.random.uniform(60, 700, count)
    shell_thick = np.random.uniform(10,200,count)
    water_thick = np.random.uniform(10,200,count)
    sld= np.random.uniform(0.1, 10.0, count)
    return_list = [{"n_shells": n_shells[i],
                    "radius": radii[i],
                    "thick_shell": shell_thick[i],
                    "thick_solvent": water_thick[i],
                    "sld": sld[i]} for i in range(count)]
    return(return_list)

def map_MLS(indict):
    params_to_map = ["n_shells", "radius", "thick_shell", "thick_solvent", "sld"]
    default_dict = {"scale":1, "background": 0.001, "sld_solvent": 0.4}
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
   

