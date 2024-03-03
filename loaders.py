#revision with highQ scaling and constant plateau addition 1-13-22
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#import rescale_exp as re

def load_spec(fn, q):
   if type(fn) == str:
    indf = pd.DataFrame(pd.read_csv(fn))
   else:
    indf = fn
   spec = np.log10(np.array(indf.loc[:,q])+1.)
   return(spec)

def load_params(fn, colnames):
   if type(fn) == str:
    indf = pd.DataFrame(pd.read_csv(fn))
   else:
    indf = fn
   outparams = {}
   #indf = pd.DataFrame(pd.read_csv(fn))
   for cn in colnames:
      if cn in indf.columns:
         outparams[cn] = np.array(indf.loc[:,cn])
      else:
         outparams[cn] = np.zeros(len(indf))
   return(outparams)

def scale(spec, maxval):
   for i in range(spec.shape[0]):
      spec[i,:] = spec[i,:]-spec[i,0]+maxval
   return(spec)

def scale_highq(spec, incoherence):
   new_spec = spec.copy()
   for i in range(spec.shape[0]):
       new_spec[i] = spec[i] -np.mean(spec[i,-2:])+incoherence
   return(new_spec)

def scale_exp(spec):
 for i in range(spec.shape[0]):
  spec[i] = re.rescale(spec[i])
 return(spec)

def shuffle(spec, params):
   targets = spec.keys()
   for t in targets:
      ss = spec[t]
      inds = np.arange(ss.shape[0])
      np.random.shuffle(inds)
      for p in params[t].keys():
         params[t][p] = params[t][p][inds]
      spec[t] = spec[t][inds]
   return(spec, params)

def load_q(datadir, qfile = 'q_200.txt'):
   q = np.loadtxt('%s/%s'%(datadir, qfile), dtype=str, delimiter=',')
   return(q)

###def load_all_spec(targets, q, datadir, dataset, prefix='train'):
###   all_spec = {}
###   maxval = 0
###   for t in targets:
###      fn = '%s/%s/%s_%s_%s.csv'%(datadir, dataset, prefix, t, dataset)
###      spec = load_spec(fn, q)
###      if np.max(spec[:,0]) > maxval:
###         maxval = np.max(spec[:,0])
###      all_spec[t] = spec
####   for t in targets:
####      all_spec[t] = scale(all_spec[t], maxval)
###   return(all_spec)

def find_bounds(sequences):
    ar_bounds = [2,4,8,16]
    shell_bounds = [.15, .4, .65, .9]
    armins = np.zeros(len(sequences))
    armaxs = np.zeros(len(sequences))
    shellmins = np.zeros(len(sequences))
    shellmaxs = np.zeros(len(sequences))
    for i in range(len(sequences)):
        s = sequences[i]
        ari = int(s[0])
        shelli = int(s[1])
        armins[i] = ar_bounds[ari]
        armaxs[i] = ar_bounds[ari+1]
        shellmins[i] = shell_bounds[shelli]
        shellmaxs[i] = shell_bounds[shelli+1]
    return(armins, armaxs, shellmins, shellmaxs)

def load_spec_range(df, q, ar_mins, ar_maxs, shell_mins, shell_maxs):
    ar = df.loc[:,'aspect_ratio']
    shell = df.loc[:, 'shell ratio']
    specs = []
    spec = load_spec(df, q)
    for i in range(ar_mins.shape[0]):
        arinds = np.logical_and(np.greater_equal(ar, ar_mins[i]), np.less(ar, ar_maxs[i]))
        shellinds = np.logical_and(np.greater_equal(shell, shell_mins[i]), np.less(shell, shell_maxs[i]))
        allowed_inds = np.where(np.logical_and(arinds, shellinds))[0]
        specs += [spec[allowed_inds]]
    print(np.concatenate(specs, axis=0).shape)
    return(np.concatenate(specs, axis=0))
    #return(np.stack(specs))

def load_params_range(df, ps, ar_mins, ar_maxs, shell_mins, shell_maxs):
    ar = df.loc[:,'aspect_ratio']
    shell = df.loc[:, 'shell ratio']
    specs = {}
    params = load_params(df, ps)
    for p in params.keys():
        specs[p] = []
    for i in range(ar_mins.shape[0]):
        arinds = np.logical_and(np.greater_equal(ar, ar_mins[i]), np.less(ar, ar_maxs[i]))
        shellinds = np.logical_and(np.greater_equal(shell, shell_mins[i]), np.less(shell, shell_maxs[i]))
        allowed_inds = np.where(np.logical_and(arinds, shellinds))[0]
        for p in params.keys():
           specs[p] += [params[p][allowed_inds]]
    outps = {}
    for p in params.keys():
        outps[p] = np.concatenate(specs[p], axis=0)
    return(outps)
    #return(np.concatenate(specs, axis=0))



def load_all_spec(targets, q, datadir, datasets, prefix='train'):
   all_spec = {}
   maxval = 0
   armins, armaxs, shellmins, shellmaxs = find_bounds(datasets)
   for t in targets:
      fn = '%s/allsas/%s_%s_allsas.csv'%(datadir, prefix, t)
      df = pd.DataFrame(pd.read_csv(fn))
      spec = load_spec_range(df, q, armins, armaxs, shellmins, shellmaxs)
      all_spec[t] = spec
   return(all_spec)

def load_all_params(targets, ps, datadir, datasets, prefix='train'):
   all_spec = {}
   maxval = 0
   armins, armaxs, shellmins, shellmaxs = find_bounds(datasets)
   for t in targets:
      all_spec[t] = {}
      fn = '%s/allsas/%s_%s_allsas.csv'%(datadir, prefix, t)
      df = pd.DataFrame(pd.read_csv(fn))
      spec = load_params_range(df, ps, armins, armaxs, shellmins, shellmaxs)
      all_spec[t] = spec
   return(all_spec)

      

def unravel_dict(spec_dict, targets = None):
    if targets is None:
        targets = spec_dict.keys()
    out_spec = spec_dict[targets[0]]
    out_labels = np.zeros(out_spec.shape[0])
    out_map = np.arange(out_spec.shape[0])
    for i in range(1,len(targets)):
        t = targets[i]
        out_spec = np.concatenate((out_spec, spec_dict[t]))
        out_labels = np.concatenate((out_labels, i*np.ones(spec_dict[t].shape[0])))
        out_map = np.concatenate((out_map, np.arange(spec_dict[t].shape[0])))
    return(out_spec, out_labels, out_map)

def concatenate_spec(spec):
   key = 0
   spec_list = []
   label_list = []
   for t in spec.keys():
      spec_list += [spec[t]]
      label_list += [key*np.ones(spec[t].shape[0])]
      key += 1
   allspec = np.concatenate(spec_list)
   labels = np.concatenate(label_list)      
   return(allspec, labels)


###def load_all_params(targets, param_list, datadir, dataset, prefix = 'train'):
###   all_params = {}
###   for t in targets:
###      fn = '%s/%s/%s_%s_%s.csv'%(datadir, dataset, prefix, t, dataset)
###      params = load_params(fn, param_list)
###      all_params[t] = params
###   return(all_params)
