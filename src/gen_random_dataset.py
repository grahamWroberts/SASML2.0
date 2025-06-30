import numpy as np
import ParamGenerator as SG
import sasmodels.data
import sasmodels.core
import sasmodels.direct_model
import VirtualInstrument
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances


def random_MLS(count):
    n_shells = np.random.randint(2, 30, count)
    radii = np.random.uniform(10, 2500, count)
    shell_thick = np.random.uniform(10,300,count)
    water_thick = np.random.uniform(10,50,count)
    sld= np.random.uniform(0.5, 5.0, count)
    background = np.random.uniform(-3.5,-1,count)
    sld_solvent = np.random.uniform(4.4,8.4,count)
    scale = np.random.uniform(0.1, 1.0, count)
    return_list = [{"n_shells": n_shells[i],
                    "radius": radii[i],
                    "thick_shell": shell_thick[i],
                    "thick_solvent": water_thick[i],
                    "background": 10**background[i],
                    "sld_solvent": sld_solvent[i],
                    "scale": scale[i],
                    "sld": sld[i]} for i in range(count)]
    return(return_list)

def random_cylinder(count):
    radii = np.random.uniform(10, 2500, count)
    length = np.random.uniform(50, 250, count)
    sld = np.random.uniform(0.5, 5.0, count)
    background = np.random.uniform(-3.5,-1,count)
    sld_solvent = np.random.uniform(4.4,8.4,count)
    scale = np.random.uniform(0.1, 1.0, count)
    return_list = [{"radius":radii[i],
                    "length":2*radii[i]+length[i],
                    "background": 10**background[i],
                    "sld_solvent": sld_solvent[i],
                    "scale": scale[i],
                    "sld":sld[i]} for i in range(count)]
    return(return_list)

def random_disk(count):
    radii = np.random.uniform(10.0, 2500.0, count)
    length = np.random.uniform(10.0, 100.0, count)
    background = np.random.uniform(-3.5,-1,count)
    sld_solvent = np.random.uniform(4.4,8.4,count)
    scale = np.random.uniform(0.1, 1.0, count)
    sld = np.random.uniform(0.5, 5.0, count)
    return_list = [{"radius":length[i]+radii[i],
                    "length":length[i],
                    "background": 10**background[i],
                    "sld_solvent": sld_solvent[i],
                    "scale": scale[i],
                    "sld":sld[i]} for i in range(count)]
    return(return_list)

def random_cs_sphere(count):
    radii = np.random.uniform(10, 2500, count)
    shell_thick = np.random.uniform(25,100,count)
    background = np.random.uniform(-3.5,-1,count)
    sld_solvent = np.random.uniform(4.4,8.4,count)
    scale = np.random.uniform(0.1, 1.0, count)
    sld= np.random.uniform(0.5, 5.0, count)
    return_list = [{"radius": radii[i],
                    "thickness": shell_thick[i],
                    "background": 10**background[i],
                    "sld_solvent": sld_solvent[i],
                    "scale": scale[i],
                    "sld_core": sld[i]} for i in range(count)]
    return(return_list)



def map_pdict(indict, default_dict):
    #params_to_map = ["n_shells", "radius", "thick_shell", "thick_solvent", "sld"]
    params_to_map = indict.keys()
    defaults = {key: default_dict[key] for key in default_dict.keys() if key not in indict.keys()}
    outdict = {key:indict[key] for key in params_to_map} | defaults
    return(outdict)


default_dict = {"scale":1,
                 "background": 0.001,
                 "sld_solvent": 6.4,
                 "radius_pd":.3,
                 "radius_pd_n":40,
                 "radius_pd_type":"schulz"}
cylinder_default_dict = {"scale":1,
                 "background": 0.001,
                 "sld_solvent": 6.4,
                 "length_pd":.3,
                 "length_pd_n":40,
                 "length_pd_type":"schulz"}
cs_sphere_default_dict = {"scale":1,
                 "background": 0.001,
                 "sld_solvent": 6.4,
                 "sld_core": 6.4,
                 "radius_pd":.3,
                 "radius_pd_n":40,
                 "radius_pd_type":"schulz"}
if __name__ == "__main__":
   count = 1000
   targets = {'cylinder': random_cylinder,
              'disk': random_disk,
              'core_shell_sphere': random_cs_sphere,
              'multilayer_vesicle': random_MLS}
   default = {'cylinder': cylinder_default_dict,
              'disk': default_dict,
              'cs_sphere': cs_sphere_default_dict,
              'multilayer_vesicle': default_dict}

   model_names = {'disk':'cylinder'}
   generators = {t:SG.ParamGenerator(targets[t], lambda x: map_pdict(x, default[t])) for t in targets.keys()}
   reference_files = ["../../libal_sas/libal/reference_sans/%s"%(s) for s in ["low_q_trimmed.ABS", "med_q_trimmed.ABS", "high_q_trimmed.ABS"]]
   vi = VirtualInstrument.VirtualInstrument()
   vi.add_references(reference_files)
   all_curves = []
   all_labels =[]
   for t in targets.keys():
      model_name = model_names[t] if t in model_names.keys() else t
      calcs, sds = vi.construct_calculators(model_name)
      t_list, n_list = generators[t].sample(count)
      print(n_list)
      print(t)
      curves = [vi.generate(model_name, kw, calcs, sds)[0] for kw in n_list]
      all_curves += curves
      all_labels += [t for i in range(count)]
   q = all_curves[0].index
   all_curves = np.array(all_curves)
   all_labels = np.array(all_labels)
   #mlv_g = SG.ParamGenerator(random_MLS, map_pdict)
   #t_list, n_list = generators['multilamelar_vesicle'].sample(5)
   #model_name = 'multilayer_vesicle'
   #print(curves)
   np.savetxt('smallX_trimmed.csv', all_curves, delimiter=',')
   np.savetxt('smally_trimmed.csv', all_labels, fmt='%s')
   np.savetxt('q_trimmed.txt', q)

   test_curves = []
   test_labels =[]
   for t in targets.keys():
      model_name = model_names[t] if t in model_names.keys() else t
      calcs, sds = vi.construct_calculators(model_name)
      t_list, n_list = generators[t].sample(count)
      print(t)
      curves = [vi.generate(model_name, kw, calcs, sds)[1] for kw in n_list]
      test_curves += curves
      test_labels += [t for i in range(count)]
   test_curves = np.array(test_curves)
   test_labels = np.array(test_labels)
   #mlv_g = SG.ParamGenerator(random_MLS, map_pdict)
   #t_list, n_list = generators['multilamelar_vesicle'].sample(5)
   #model_name = 'multilayer_vesicle'
   #print(curves)
   np.savetxt('testX_trimmed.csv', test_curves, delimiter=',')
   np.savetxt('testy_trimmed.csv', test_labels, fmt='%s')

   for l in np.unique(all_labels):
       all_valid = np.where(all_labels==l)[0]
       test_valid = np.where(test_labels==l)[0]
       dp = pairwise_distances(all_curves[all_valid], test_curves[test_valid])
       print(dp)
       print(np.min(dp))

                    

   

