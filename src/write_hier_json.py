import numpy as np
import json

targets = ['cylinder', 'disk', 'sphere', 'cs_cylinder', 'cs_disk', 'cs_sphere']


def parse_ss(s):
    vals = s.split('_')
    ttype = vals[0]
    c = float(vals[1])
    gamma = float(vals[2])/200.
    kernel = vals[3]
    coef0 = float(vals[-1])
    outdict = {'type': ttype,
               'c': c,
               'gamma': gamma,
               'kernel': {'type': kernel},
               'coeff0': coef0}
    return(outdict)

def write_hm_dict(i):
    if isinstance(hierarchical_map[i][0], str):
        leftLab = targets[int(float(hierarchical_map[i][0]))]
        classLeft = [leftLab]
        left = {'class': leftLab}
    else:
        left = write_hm_dict(hierarchical_map[i][0])
        classLeft = left['classLeft'] + left['classRight']
    if isinstance(hierarchical_map[i][1], str):
        rightLab = targets[int(float(hierarchical_map[i][1]))]
        classRight = [rightLab]
        right = {'class': rightLab}
    else:
        right = write_hm_dict(hierarchical_map[i][1])
        classRight = right['classLeft'] + right['classRight']
    outdict = { 'classifier': struct_dicts[i],
                'classLeft': classLeft,
                'classRight': classRight,
                'left': left,
                'right': right,
                'jobfile': 'classifier_%d.joblib'%(i)}
    return(outdict)

           
decision1 = {0:0,1:0,2:1,3:0,4:0,5:1}
decision2 = {2:0,5:1}
decision3 = {0:0,1:0,3:1,4:1}
decision4 = {0:0,1:1}
decision5 = {3:0,4:1}
decisions = [decision1, decision2, decision3, decision4, decision5]
hierarchical_map = [{0:2,1:1},{0:'2',1:'5'},{0:3,1:4},{0:'0',1:'1'},{0:'3',1:'4'}]
ss = ["svc_100.000000_10.000000_rbf_1.000000",
      "svc_10.000000_100.000000_rbf_1.000000",
      "svc_10.000000_100.000000_rbf_1.000000",
      "svc_10000.000000_1.000000_rbf_1.000000",
      "svc_100.000000_1.000000_rbf_1.000000"]

struct_dicts = [parse_ss(s) for s in ss]
for s in struct_dicts:
    print(s)
outdict = write_hm_dict(0)
print(outdict)

outfile = open('ex_hierarchical.json', 'w')
json_str = json.dumps(outdict, indent = 3)
outfile.write(json_str)



