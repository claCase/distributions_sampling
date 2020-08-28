import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from numpy.random import multivariate_normal as mv
import neural_network
import distributions as distr


distr.circular_manifold_sampling()
distr.mixture_model()
distr.multivariate_mixture_sampling()
distr.quadratic_manifold()

modes = 11
n_per_mode = 50
var = 5
r = 20
data, cos_mu, sin_mu = distr.generate_circular_multimodal(modes=modes, r=r, n_per_mode=n_per_mode, var=var)
fg, ax = plt.subplots(1, 1)
for i in range(0, data.shape[0], n_per_mode):
    ax.scatter(data[i:i + n_per_mode, 0], data[i:i + n_per_mode, 1])
ax.scatter(cos_mu, sin_mu, color="red")
plt.show()
