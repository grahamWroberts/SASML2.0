import numpy as np
import ParamGenerator as SG
import sasmodels.data
import sasmodels.core
import sasmodels.direct_model
import VirtualInstrument
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
import pandas as pd
import xarray as xr

def random_teubner(count):
    scale = np.random.uniform(0.1, 1.0, count)
    background = np.random.uniform(-3.5,-1,count)
    background = np.random.uniform(0.001, 0.001, count)
    volfraction_a = np.random.uniform(0.1, 0.9, count)
    sld_a = np.random.uniform(4.4,8.4,count)
    sld_b = np.random.uniform(4.4,8.4,count)
    d = np.random.uniform(50, 500, count)
    xi = np.random.uniform(10,100,count)
    return_list = [{"scale": scale[i],
                    "background":background[i],
                    "volfraction_a":volfraction_a[i],
                    "sld_a":sld_a[i],
                    "sld_b":sld_b[i],
                    "d":d[i],
                    "xi":background[i]} for i in range(count)]
    return(return_list)

def random_surf_frac(count):
    scale = np.random.uniform(0.1, 1.0, count)
    background = np.random.uniform(-3.5,-1,count)
    background = np.random.uniform(0.001, 0.001, count)
    radius = np.random.uniform(10, 1000, count)
    fractal_dim_surf = np.random.uniform(1,10,count)
    cutoff_length = np.random.uniform(50,1000,count)
    return_list = [{"scale": scale[i],
                    "background":background[i],
                    "radius":radius[i],
                    "fractal_dim_surf":fractal_dim_surf[i],
                    "cutoff_length":cutoff_length[i]} for i in range(count)]
    return(return_list)

def random_mass_fractal(count):
    scale = np.random.uniform(0.1, 1.0, count)
    background = np.random.uniform(-3.5,-1,count)
    background = np.random.uniform(0.001, 0.001, count)
    radius = np.random.uniform(10, 1000, count)
    fractal_dim_mass = np.random.uniform(1,10,count)
    cutoff_length = np.random.uniform(50,500,count)
    return_list = [{"scale": scale[i],
                    "background":background[i],
                    "radius":radius[i],
                    "fractal_dim_mass":fractal_dim_mass[i],
                    "cutoff_length":cutoff_length[i]} for i in range(count)]
    return(return_list)

    

def random_MLS(count):
    n_shells = np.random.randint(10, 30, count)
    radii = np.random.uniform(100, 500, count)
    shell_thick = np.random.uniform(30,60,count)
    water_thick = np.random.uniform(10,20,count)
    sld= np.random.uniform(0.5, 5.0, count)
    background = np.random.uniform(-3.5,-1,count)
    background = np.random.uniform(0.001, 0.001, count)
    sld_solvent = np.random.uniform(4.4,8.4,count)
    pds = np.random.uniform(0.3, 0.5, count)
    scale = np.random.uniform(0.1, 1.0, count)
    return_list = [{"n_shells": n_shells[i],
                    "radius": radii[i],
                    "thick_shell": shell_thick[i],
                    "thick_solvent": water_thick[i],
                    "background": 10**background[i],
                    "sld_solvent": sld_solvent[i],
                    "scale": scale[i],
                    "radius_pd": pds[i],
                    "radius_pd_type": "schulz",
                    "radius_pd_n": 8,
                    "radius_pd_nsigma": 30,
                    "sld": sld[i]} for i in range(count)]
    return(return_list)

def random_cylinder(count):
    radii = np.random.uniform(10, 2500, count)
    length = np.random.uniform(50, 250, count)
    sld = np.random.uniform(0.5, 5.0, count)
    background = np.random.uniform(-3.5,-1,count)
    background = np.random.uniform(0.001, 0.001, count)
    sld_solvent = np.random.uniform(4.4,8.4,count)
    pds = np.random.uniform(0.3, 0.5, count)
    scale = np.random.uniform(0.1, 1.0, count)
    return_list = [{"radius":radii[i],
                    "length":2*radii[i]+length[i],
                    "background": 10**background[i],
                    "sld_solvent": sld_solvent[i],
                    "scale": scale[i],
                    "sld":sld[i]} for i in range(count)]
    return(return_list)

def random_disk(count):
    radii = np.random.uniform(50.0, 500.0, count)
    length = np.random.uniform(30.0, 60.0, count)
    background = np.random.uniform(-3.5,-1,count)
    background = np.random.uniform(0.001, 0.001, count)
    sld_solvent = np.random.uniform(4.4,8.4,count)
    scale = np.random.uniform(0.1, 1.0, count)
    pds = np.random.uniform(0.3, 0.5, count)
    sld = np.random.uniform(0.5, 5.0, count)
    return_list = [{"radius":length[i]+radii[i],
                    "length":length[i],
                    "background": 10**background[i],
                    "sld_solvent": sld_solvent[i],
                    "scale": scale[i],
                    "radius_pd": pds[i],
                    "radius_pd_type": "schulz",
                    "radius_pd_n": 8,
                    "radius_pd_nsigma": 30,
                    "sld":sld[i]} for i in range(count)]
    return(return_list)

def random_cs_sphere(count):
    radii = np.random.uniform(100, 1000, count)
    shell_thick = np.random.uniform(30,60,count)
    background = np.random.uniform(-3.5,-1,count)
    background = np.random.uniform(0.001, 0.001, count)
    sld_solvent = np.random.uniform(4.4,8.4,count)
    scale = np.random.uniform(0.1, 1.0, count)
    sld= np.random.uniform(0.5, 5.0, count)
    pds = np.random.uniform(0.3, 0.5, count)
    return_list = [{"radius": radii[i],
                    "thickness": shell_thick[i],
                    "background": 10**background[i],
                    "sld_solvent": sld_solvent[i],
                    "scale": scale[i],
                    "radius_pd": pds[i],
                    "radius_pd_type": "schulz",
                    "radius_pd_n": 8.,
                    "radius_pd_nsigma": 30,
                    "sld_core": sld[i]} for i in range(count)]
    return(return_list)

def random_wormlike_micelle(count):
    length = np.random.uniform(3500, 3500, count)
    radii = np.random.uniform(30, 60, count)
    background = np.random.uniform(-3.5, -1, count)
    background = np.random.uniform(0.001, 0.001, count)
    kuhn_length = np.random.uniform(50, 50, count)
    pds = np.random.uniform(0.3, 0.5, count)
    axis_ratio = np.random.uniform(1.0, 2.0, count)
    return_list = [{"radius":radii[i],
                    "length":length[i],
                    "background":10**background[i],
                    "kuhn_length":kuhn_length[i],
                    "radius_pd": pds[i],
                    "radius_pd_type": "schulz",
                    "radius_pd_n": 8.,
                    "radius_pd_nsigma": 30,
                    "axis_ratio":axis_ratio[i]} for i in range(count)]
    return(return_list)



def map_pdict(indict, default_dict):
    #params_to_map = ["n_shells", "radius", "thick_shell", "thick_solvent", "sld"]
    #params_to_map = indict.keys()
    defaults = {key: default_dict[key] for key in default_dict.keys() if key not in indict.keys()}
    #outdict = {key:indict[key] for key in params_to_map} | defaults
    outdict = indict|defaults
    return(outdict)

def filter_nan(arr):
    return(np.where(np.all(np.logical_not(np.isnan(arr)), axis=1))[0])

def tabularize_params(param_table):
    keys = param_table[0].keys()
    lim = len(param_table)
    outdf = {k:np.array([p[k] for p in param_table]) for k in keys}
    return(pd.DataFrame(outdf))

def make_map_function(reference_dict):
    def newfunc(indict):
        return(map_pdict(indict, reference_dict))
    return(newfunc)

def concat_params(labels, params):
    unique_params = []
    ptypes = {}
    for morphology in params.keys():
        for param in params[morphology].keys():
            if param not in unique_params:
                unique_params += [param]
                ptypes[param] = type(params[morphology][param][0])
    all_params = {param: np.zeros(len(labels), dtype = ptypes[param]) for param in unique_params}
    for morphology in params.keys():
        #inds = np.where(labels == morphology)[0]
        inds = np.where(np.array([lab == morphology for lab in labels]))[0]
        for param in unique_params:
            if param in params[morphology].keys():
                all_params[param][inds] = params[morphology][param]
            else:
                print(inds)
                all_params[param][inds] = np.nan * np.ones(inds.shape[0])
    return(all_params)

def unwrap_params(params):
    unique_params = params[0].keys()
    params = {p: np.array([para[p] for para in params]) for p in unique_params}
    return(params)

def make_xarray(curves, params, labels, q, test_curves = None, test_params = None, test_labels = None):
    new_array = xr.Dataset()
    inds = np.arange(labels.shape[0])
    test_inds = np.arange(test_labels.shape[0])
    new_array['sample'] = inds
    new_array['q'] = q
    new_array['SAS_curves'] = (['sample', 'q'], curves)
    new_array['labels'] = (['sample'], labels)
    for p in params.keys():
        new_array[p] = (['sample'], params[p])
    if test_curves is not None and test_params is not None and test_labels is not None:
        new_array['test_sample'] = test_inds
        new_array['test_curves'] = (['test_sample', 'q'], test_curves)
        new_array['test_labels'] = (['test_sample'], test_labels)
        for p in test_params.keys():
            new_array["test_%s"%(p)] = (['test_sample'], params[p])
    return(new_array)



general_default_dict = {"scale":1,
                 "background": 0.001}
###                 "sld_solvent": 6.4,
###                 "radius_pd":.3,
###                 "radius_pd_n":40,
###                 "radius_pd_type":"schulz"}
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
###   targets = {'cylinder': random_cylinder,
###              'disk': random_disk,
###              'core_shell_sphere': random_cs_sphere,
###              'multilayer_vesicle': random_MLS}
###   default = {'cylinder': cylinder_default_dict,
###              'disk': default_dict,
###              'cs_sphere': cs_sphere_default_dict,
###              'multilayer_vesicle': default_dict}
   targets = {'flexible_cylinder_elliptical': random_wormlike_micelle,
              'core_shell_sphere': random_cs_sphere,
              'multilayer_vesicle': random_MLS,
              'disk': random_disk}
   default_map = {'flexible_cylinder_elliptical': general_default_dict,
              'core_shell_sphere': cs_sphere_default_dict,
              'multilayer_vesicle': general_default_dict,
              'disk': cylinder_default_dict}

   model_names = {'disk':'cylinder'}
   parameterizers = {t: make_map_function( default_map[t]) for t in targets.keys()}

   generators = {t:SG.ParamGenerator(targets[t], parameterizers[t]) for t in targets.keys()}
   #generators = {t:SG.ParamGenerator(targets[t], lambda x: map_pdict(x, default_map[t])) for t in targets.keys()}
#   reference_files = ["../../libal_sas/libal/reference_sans/%s"%(s) for s in ["low_q_trimmed.ABS", "med_q_trimmed.ABS", "high_q_trimmed.ABS"]]
   reference_files = ["../Long_Config_r28189_ResBj_1mm_1D_combined.txt", "../Short_Config_r27386_ResB_Bj_1D_combined.txt"]
   vi = VirtualInstrument.VirtualInstrument()
   vi.add_references(reference_files)
   all_curves = []
   all_labels =[]
   all_params = {}
   for t in targets.keys():
      t_list, n_list = None, None
      model_name = model_names[t] if t in model_names.keys() else t
      calcs, sds = vi.construct_calculators(model_name)
      t_list, n_list = generators[t].sample(count)
      all_params[t] = unwrap_params(t_list)
      param_df = tabularize_params(n_list)
      param_df.to_csv("parameters_%s.csv"%(t))
      curves = [vi.generate(model_name, kw, calcs, sds)[0] for kw in n_list]
      all_curves += curves
      all_labels += [t for i in range(count)]
   param_lists = concat_params(all_labels, all_params)
   q = all_curves[0].index
   all_curves = np.array(all_curves)
   all_valid = filter_nan(all_curves)
   all_curves = all_curves[all_valid]
   all_labels = np.array(all_labels)
   all_labels = all_labels[all_valid]
   for p in param_lists.keys():
       param_lists[p] = param_lists[p][all_valid]
   #mlv_g = SG.ParamGenerator(random_MLS, map_pdict)
   #t_list, n_list = generators['multilamelar_vesicle'].sample(5)
   #model_name = 'multilayer_vesicle'
   #print(curves)
   np.savetxt('smallX_ex.csv', all_curves, delimiter=',')
   np.savetxt('smally_ex.csv', all_labels, fmt='%s')
   np.savetxt('q_trimmed.txt', q)

   test_curves = []
   test_labels =[]
   test_params = {}
   for t in targets.keys():
      model_name = model_names[t] if t in model_names.keys() else t
      calcs, sds = vi.construct_calculators(model_name)
      t_list, n_list = generators[t].sample(count)
      test_params[t] = unwrap_params(t_list)
      param_df = tabularize_params(n_list)
      param_df.to_csv("test_parameters_%s.csv"%(t))
      curves = [vi.generate(model_name, kw, calcs, sds)[1] for kw in n_list]
      test_curves += curves
      test_labels += [t for i in range(count)]
   test_param_lists = concat_params(test_labels, test_params)
   test_curves = np.array(test_curves)
   test_valid = filter_nan(test_curves)
   test_curves = test_curves[test_valid]
   test_labels = np.array(test_labels)
   test_labels = test_labels[test_valid]
   for p in param_lists.keys():
       param_lists[p] = param_lists[p][all_valid]
   new_xr = make_xarray(all_curves, param_lists, all_labels, q, test_curves, test_param_lists, test_labels)
   new_xr.to_netcdf("data.nc")
   #mlv_g = SG.ParamGenerator(random_MLS, map_pdict)
   #t_list, n_list = generators['multilamelar_vesicle'].sample(5)
   #model_name = 'multilayer_vesicle'
   #print(curves)
   np.savetxt('testX_ex.csv', test_curves, delimiter=',')
   np.savetxt('testy_ex.csv', test_labels, fmt='%s')
   print("all_curves nan %d test nan %d"%(np.sum(np.isnan(all_curves)), np.sum(np.isnan(test_curves))))

   for l in np.unique(all_labels):
       all_valid = np.where(all_labels==l)[0]
       test_valid = np.where(test_labels==l)[0]
       dp = pairwise_distances(all_curves[all_valid], test_curves[test_valid])

                    

   

