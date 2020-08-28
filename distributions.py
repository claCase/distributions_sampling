import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
import seaborn as sns

def circular_manifold_sampling():
    n = 1000
    theta = np.linspace(0, 2 * np.pi, 360)
    x_axs, y_axs = np.cos(theta), np.sin(theta)
    r = np.random.uniform(0, 1, n)
    theta = np.random.uniform(0, 2 * np.pi, n)
    x = np.cos(theta) * r
    y = np.sin(theta) * r

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y)
    plt.plot(x_axs, y_axs, color="red")
    plt.show()

def quadratic_manifold():
    # Quadratic manifold sampling
    n = 1000
    x = np.random.normal(0, 1, n)
    y = np.square(x) + np.random.sample(n) * np.random.uniform(0,1, n)
    plt.scatter(x, y)
    plt.show()


def generate_circular_multimodal(modes, r, n_per_mode, var):
    cos_mu = np.cos(np.linspace(0, 2 * np.pi, modes + 1)) * r
    cos_mu = cos_mu[:-1]
    sin_mu = np.sin(np.linspace(0, 2 * np.pi, modes + 1)) * r
    sin_mu = sin_mu[:-1]
    data = np.empty((1, 2))
    for c, s in zip(cos_mu, sin_mu):
        points = multivariate_normal([c, s], np.asarray([var, 0, 0, var]).reshape((2, 2))).rvs(size=n_per_mode)
        data = np.concatenate((data, points), axis=0)
    return np.asarray(data[1:]), cos_mu, sin_mu


def mixture_model():
    # MIXTURE MODEL
    from scipy.stats import norm, multivariate_normal
    from matplotlib.widgets import Slider
    n_samples = 10000
    input_dim = 1
    output_dim = 1
    hidden_size = 4
    activation = "relu"
    p = 0.6
    mu1 = 2
    mu2 = 15
    sd1 = 2
    sd2 = 5
    lower = -10
    upper = 30
    bins = 150
    # DENSITY FUNCTION
    x = np.linspace(lower, upper, bins)
    normal1 = norm.pdf(x, mu1, sd1)
    normal2 = norm.pdf(x, mu2, sd2)

    def normal_mix(val):
        return val * normal1 + (1 - val) * normal2

    def update(val):
        fig.canvas.draw_idle()

        ax.clear()
        ax.plot(x, normal_mix(val))

    # multi_norm = multivariate_normal.pdf((x, x), np.asarray(mu1*p, 0, 0, mu2*(1-p)).reshape(2,2), np.asarray(sd1*np.sqrt(p), 0, 0, sd2*np.sqrt(1-p)).reshape(2,2))
    # RANDOM SAMPLE
    mode_choice = np.random.uniform(0, 1, n_samples) > p
    mode_choice = np.vstack((1 - mode_choice, mode_choice)).T
    # print(mode_choice.shape, mode_choice[:3])
    X = np.vstack((np.random.normal(mu1, sd1, n_samples), np.random.normal(mu2, sd2, n_samples))).T
    X = (X * mode_choice).flatten()
    X = X[X != 0]

    fig, ax = plt.subplots(figsize=(8, 10))
    new_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    slider = Slider(new_ax, "prob", 0, 1)
    ax.hist(X, bins=bins, range=(lower, upper), density=True)
    ax.plot(x, normal_mix(p))

    slider.on_changed(update)
    plt.show()


def multivariate_mixture_sampling():
    # MULTIVARIATE MIX DENSITY
    n_samples = 10000
    input_dim = 1
    output_dim = 1
    hidden_size = 4
    activation = "relu"

    p = 0.4
    mu1 = [10, 8]
    mu2 = [15, 15]
    sd1 = np.asarray([5, 0, 0, 2]).reshape(2, 2)
    sd2 = np.asarray([5, 0, 0, 1]).reshape(2, 2)
    x_lim = [0, 30]
    y_lim = [0, 30]
    bins = 100

    # DENSITY FUNCTION
    def normal_mix(val, normal1, normal2):
        return val * normal1 + (1 - val) * normal2

    x = np.linspace(0, 30, bins)
    x, y = np.meshgrid(x, x)
    pos = np.dstack((x, y))
    mv1 = multivariate_normal(mean=mu1, cov=sd1)
    mv2 = multivariate_normal(mean=mu2, cov=sd2)
    normal1 = mv1.pdf(pos)
    normal2 = mv2.pdf(pos)
    normal_mix_pdf = normal_mix(p, normal1, normal2)
    normal1_sample = mv1.rvs(size=n_samples)
    normal2_sample = mv2.rvs(size=n_samples)
    # multi_norm = multivariate_normal.pdf((x, x), np.asarray(mu1*p, 0, 0, mu2*(1-p)).reshape(2,2), np.asarray(sd1*np.sqrt(p), 0, 0, sd2*np.sqrt(1-p)).reshape(2,2))
    # RANDOM SAMPLE
    mode_choice = np.random.uniform(0, 1, n_samples) < p
    mode_choice = np.vstack((mode_choice, mode_choice))
    mode_choice = np.vstack((mode_choice, 1 - mode_choice)).T
    # print(mode_choice.shape, mode_choice[:3])
    # X = np.vstack((np.random.normal(mu1, sd1, n_samples), np.random.normal(mu2, sd2, n_samples))).T
    X = np.hstack((normal1_sample, normal2_sample))
    X = (X * mode_choice).reshape(-1, 2)
    X_keep = (X != 0)
    X = X[X_keep].reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.scatter(X[:, 0], X[:, 1], color="red", s=0.1)
    hist = ax.hist2d(X[:, 0], X[:, 1], bins=bins, density=True)
    fig.colorbar(hist[3], ax=ax)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=.1)
    plt.contour(x, y, normal_mix_pdf, linewidths=3)

    df = pd.DataFrame(X, columns=["X", "Y"])
    sns.jointplot("X", "Y", data=df, kind="kde", space=0, color="b")
    plt.show()
