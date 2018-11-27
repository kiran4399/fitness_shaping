import numpy as np
import json
import copy
import os
params = 867


def load_weight(filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    return np.array(data[0])



def savejson(weights, base, iteration):
    filenames = []
    for each in reversed(weights):
        filenames.append(base + str(iteration) + '.json')
        with open(base + str(iteration) + '.json', 'wt') as out:
            res = json.dump([each.flatten().tolist()], out, sort_keys=True, indent=0, separators=(',', ': '))
        iteration += 100
    return filenames



def gensamples(filename, numsamples):
    mu = load_weight(filename)
    sigma = 0.1*np.eye(len(mu))

    samples = np.random.multivariate_normal(mu,sigma, numsamples)

    savejson(samples, 'log/human/', 0)

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

gensamples('log/carracing.cma.16.64.json', 64)

'''
samples = sorted(absoluteFilePaths('log/samples/'))
newmean = load_weight('log/experiment/400.json')
mu = load_weight('log/experiment/300.json')
newweight = copy.deepcopy(samples)
constants = copy.deepcopy(samples)
for i in range(len(samples)):
    newweight[i] = (load_weight(samples[i]) - newmean)
    #print(newweight[i].shape)
    constants[i] = np.mean(newweight[i]/load_weight(samples[i]))
    print(newweight[i])
print(constants)
savejson(newweight, 'log/newsamples/', 0)
'''