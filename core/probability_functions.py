import numpy as np
from scipy.stats import norm


# Example initialization function for ParticleFilter class, kwargs would include std
def sampleNormalDistribution(std):
    m = np.zeros(std.shape)
    sample = norm.rvs(m, std)
    prob_individual = norm.pdf(sample, m, std)
    prob_individual[np.isnan(prob_individual)] = 1.0
    return sample, np.prod(prob_individual)


# Example motion model function for Particle Filter class, kwargs would include std
def additiveGaussianNoise(state, std):
    sample, prob = sampleNormalDistribution(std)
    return state + sample, prob
