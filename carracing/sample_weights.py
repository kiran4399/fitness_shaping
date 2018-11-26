import numpy as np
import json

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

    savejson(samples, 'log/samples/', 0)



gensamples('log/experiment/300.json', 64)
