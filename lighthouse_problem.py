'''
The lighthouse problem: an application of nested sampling.

The intent of this code is to reimplement in Python the inference 
of the light house location using the nested sampling algorithm. 
It is adapted from the code provided by Sivia and Skilling in the 
second edition of their book "Data Analysis: A Bayesian Tutorial".

It is written to closely follow the original C-listing, but deviating
in those aspects where sticking closely to the C-listing would yield
very non-pythonic code.

As the original C-listings are released under GNU General Public License
it seems fair to release this code under the same license.

License: https://www.gnu.org/licenses/gpl-3.0.en.html

Author: Ronald A.J. van Elburg, RonaldAJ__at__vanElburg.eu
Copyright: Hanze UAS, Groningen, the Netherlands
'''

from nested_sampling import NestedSampler, modelSample
from functools import partial
from numpy import random
import numpy as np

doplot = True

# Set the measurement data
x_measurements = [
    4.73, 0.45, -1.73, 1.09, 2.19, 0.12, 1.31, 1.00, 1.32, 1.07, 0.86,
    -0.49, -2.59, 1.73, 2.11, 1.61, 4.98, 1.71, 2.23, -57.20, 0.96, 1.25,
    -1.56, 2.45, 1.19, 2.17, -10.66, 1.91, -4.16, 1.92, 0.10, 1.98, -2.51,
    5.55, -0.47, 1.91, 0.95, -0.78, -0.84, 1.72, -0.01, 1.48, 2.70, 1.21,
    4.41, -4.79, 1.33, 0.81, 0.20, 1.58, 1.29, 16.19, 2.75, -2.38, -1.79,
    6.50, -18.53, 0.72, 0.94, 3.64, 1.94, -0.11, 1.57, 0.57]

class measurementSample():

    def __init__(self, instance):
        self.instance = instance

    def __repr__(self):
        return self.instance.__repr__()


measurement_data = [measurementSample({'x': x}) for x in x_measurements]

# Define the loglikelihood function


def loglikelihood(measurements, model_pars):
    loglikelihood = 0
    for measurement in measurements:
        loglikelihood += np.log(model_pars['y']/np.pi)-np.log((measurement.instance['x']-model_pars['x'])**2+model_pars['y']**2)
    return loglikelihood

myloglikelihood = partial(loglikelihood, measurement_data)

# Initialize a set of samples from the prior distribution
N = 100
random.seed(2)

u_list = random.random_sample(N)
x_list = 4.0*u_list - 2.0

v_list = random.random_sample(N)
y_list = 2.0*v_list

prior_distributions = [modelSample({'u': u, 'x': x, 'v': v, 'y': y}, myloglikelihood) for (u, x, v, y) in zip(u_list, x_list, v_list, y_list)]


def Evolve(modelsample, logLstar):
    ''' Function to evolve an object within the likelihood constraint (analogous top Explore in original listing)
    '''
    step = 0.1
    accept = 0
    reject = 0
    evolvedsample = modelsample

    for dummyindex in range(20):
        u = evolvedsample.model_pars['u'] + step * (2.*random.random_sample() - 1.)
        u -= np.floor(u)

        v = evolvedsample.model_pars['v'] + step * (2.*random.random_sample() - 1.)
        v -= np.floor(v)

        x = 4.0*u - 2.0
        y = 2.0*v

        trial_sample = modelSample({'u': u, 'x': x, 'v': v, 'y': y}, myloglikelihood)

        if trial_sample.logL > logLstar:
            evolvedsample = trial_sample
            accept += 1
        else:
            reject += 1

        if accept > reject:
            step *= np.exp(1.0/accept)
        elif reject > accept:
            step /= np.exp(1.0/reject)

    return evolvedsample

NS = NestedSampler(prior_distributions, 1000,  Evolve)
sampled_distribution, sampled_posterior = NS.mainloop()

# Prepare a plot of the sampled prior
if doplot:
    from matplotlib import pyplot as plt

    for distribution in [sampled_distribution, sampled_posterior, prior_distributions]:
        prior_x = [sample.model_pars['x'] for sample in distribution]
        prior_y = [sample.model_pars['y'] for sample in distribution]

        plt.figure()
        plt.plot(prior_x, prior_y, '+')
        plt.show()

    print(myloglikelihood({'x': 0, 'y': 2}))
    print(measurement_data)
    print(prior_distributions)
