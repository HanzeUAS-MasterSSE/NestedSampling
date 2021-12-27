'''
Nested sampling in Python. (Not optimized for speed!)

The intent of this code is to reimplement in Python the nested
sampling algorithm as provided by Sivia and Skilling in the
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
import numpy as np
from  numpy import random
import sys

class NestedSampler():

    def __init__(self, sampled_prior, max_iterations, evolvefie):
        self.sampled_posterior = list()
        self.sampled_prior = sampled_prior
        self.sampled_distribution = self.sampled_prior.copy()
        self.max_iterations = max_iterations
        self.prior_size = len(self.sampled_prior)
        self.logwidth = np.log(1.0 - np.exp(-1.0/self.prior_size))

        self.H = 0
        self.logLstar = None
        self.logZ = -sys.float_info.max

        self.evolvefie = evolvefie

    def find_worst_object(self, sample_list):
        index_worst = 0
        worst_sample = sample_list[0]

        for index, sample in enumerate(sample_list):
            if sample.logL < worst_sample.logL:
                worst_sample = sample
                index_worst = index

        return index_worst, worst_sample

    def mainloop(self):
        for dummy_index in range(self.max_iterations):

            # find worst object
            selected_for_replacement, worst_sample = self.find_worst_object(self.sampled_distribution)
            worst_sample.logWt = self.logwidth + worst_sample.logL

            # update evidence and information
            if self.logZ > worst_sample.logWt:
                logZnew = self.logZ + np.log(1 + np.exp(worst_sample.logWt - self.logZ))
            else:
                logZnew = worst_sample.logWt + np.log(1 + np.exp(self.logZ - worst_sample.logWt))

            H = np.exp(worst_sample.logWt - logZnew)
            H *= worst_sample.logL
            H += np.exp(self.logZ - logZnew)*(self.H + self.logZ) - logZnew

            self.H = H
            self.logZ = logZnew

            # save worst sample
            self.sampled_posterior.append(worst_sample)

            # copy and evolve other modelsample
            copy = selected_for_replacement
            while copy == selected_for_replacement:
                copy = random.randint(0, self.prior_size)

            self.logLstar = worst_sample.logL

            newModelSample = self.evolvefie(self.sampled_distribution[copy], self.logLstar)
            self.sampled_distribution[selected_for_replacement] = newModelSample  # Replace worst sample by either copy or its evolved variant

            self.logwidth = -1.0/self.prior_size

        return  self.sampled_distribution, self.sampled_posterior
        
'''
For the nested sampling algorithm the prior is represented by drawing 
n_objects samples from the prior distribution. So we need to have 
objects representing these samples , which have the required methods 
for use with the algorithm.

If we look at the c-listing in chapter 9 we see that the following 
member variables are accessed for each object:
    logL:  logLikelihood = ln( P({data}| {model parametrs}), calculated at instantiation
    logWt: log(weight) , calculated by nested sampling algorithm
    in addition there are model parameters:
    model_pars
    
'''

class modelSample():

    def __init__(self, model_pars, myloglikelihood):
        self.model_pars = model_pars
        self.logL = myloglikelihood(model_pars)
        self.logWt = None

    def __repr__(self):
        return self.model_pars.__repr__()


