import numpy as np
def random(count):
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

def map(indict):
    params_to_map = ["n_shells", "radius", "thick_shell", "thick_solvent", "sld"]
    default_dict = {"scale":1, "background": 0.001, "sld_solvent": 0.4}
    outdict = {key:indict[key] for key in params_to_map} | default_dict
    return(outdict)
