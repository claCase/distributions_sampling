import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from numpy.random import multivariate_normal as mv


parser = argparse.ArgumentParser()
parser.add_argument("--distr", type=str, default="uniform")
parser.add_argument("--n", type=int, default=1000)
args = parser.parse_args()
distr = args.distr
n = args.n

def nn():
    # import tensorflow.keras as k
    def sigmoid(y):
        output = 1 / (1 + np.exp(y))
        return output


    def sigmoid_derivative(y):
        s = sigmoid(y)
        return (1 - s) * s


    def softmax(z):
        # https://stackoverflow.com/a/39558290
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div


    def softmax_derivative(z):
        # https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
        s = softmax(z)
        diag = np.diag(s)
        s = s[:, np.newaxis]
        tiled = np.tile(s, s.shape[0])
        ds = diag - tiled.dot(tiled.T)
        return ds


    def sum_squares(out, label):
        print(out.shape, label.shape)
        return np.sum(np.square((out - label)))


    def cross_entropy(predictions, targets, epsilon=1e-12):
        # https://stackoverflow.com/questions/47377222/what-is-the-problem-with-my-implementation-of-the-cross-entropy-function
        """
        Computes cross entropy between targets(encoded as one-hot vectors)and predictions.
        Input: predictions (N, k) ndarray
                targets (N, k) ndarray
        Returns: scalar
        """
        N = predictions.shape[0]
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
        print(ce.shape)
        return ce


    data_length = 300
    mu = -3  # media True
    var = 1  # varianza True
    mu2 = 3  # media False
    var2 = 2  # varianza False

    input_size = 2
    hidden_size = 5
    output_size = 2

    # DATA and LABELS
    a_True = np.random.normal(mu, var, (data_length, input_size))
    labels_True = np.ones((data_length, 1))
    labels_True = np.hstack((labels_True, np.zeros((data_length, 1))))
    a_False = np.random.normal(mu2, var2, (data_length, input_size))
    labels_False = np.zeros((data_length, 1))
    labels_False = np.hstack((labels_False, np.ones((data_length, 1))))
    a_tot = np.vstack((a_True, a_False))
    l_tot = np.vstack((labels_True, labels_False))

    # NETWORK: 2 layer feed foward neural network
    W = np.random.normal(0, 1, (input_size, hidden_size))
    b = np.random.normal(0, 1, hidden_size)
    Y = np.matmul(a_tot, W) + b
    Z = sigmoid(Y)

    W2 = np.random.normal(0, 1, (hidden_size, output_size))
    b2 = np.random.normal(0, 1, output_size)
    Y2 = np.matmul(Z, W2) + b2
    Z2 = softmax(Y2)  # output

    # PLOT Outputs
    fg = plt.figure(figsize=(25, 5))
    x, y = np.meshgrid(np.linspace(mu - 3, mu + 3, 50), np.linspace(mu - 3, mu + 3, 50))
    mu = np.asarray((mu, mu))
    cov = np.asarray((var, 0, 0, var)).reshape((2, 2))
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x;
    pos[:, :, 1] = y
    z = multivariate_normal(mu, cov)

    x2, y2 = np.meshgrid(np.linspace(mu2 - 3, mu2 + 3, 50), np.linspace(mu2 - 3, mu2 + 3, 50))
    mu2 = np.asarray((mu2, mu2))
    cov2 = np.asarray((var2, 0, 0, var2)).reshape((2, 2))
    pos2 = np.empty(x2.shape + (2,))
    pos2[:, :, 0] = x2;
    pos2[:, :, 1] = y2
    z2 = multivariate_normal(mu2, cov2)

    # Plot data and gaussian density
    # ax = fg.add_subplot(1,3,1, projection = "3d")
    # ax.view_init(60, 35)
    ax = fg.add_subplot(1, 4, 1)
    ax.contour(x, y, z.pdf(pos), cmap="GnBu", linewidths=2)
    ax.contour(x2, y2, z2.pdf(pos2), cmap="OrRd", linewidths=2)
    ax.scatter(a_tot[:data_length, 0], a_tot[:data_length, 1], color="blue", label="Si")
    ax.scatter(a_tot[data_length:, 0], a_tot[data_length:, 1], color="red", label="No")
    # ax.plot_surface(x,y,z.pdf(pos), cmap="GnBu")
    # ax.plot_surface(x2,y2,z2.pdf(pos2), cmap="OrRd",)
    ax.set_title("Data points")
    ax.legend(loc=2)

    # Hidden Linear Outout plot
    ax = fg.add_subplot(1, 4, 2)
    ax.scatter(Y[:data_length, 0], Y[:data_length, 1], color="blue")
    ax.scatter(Y[data_length:, 0], Y[data_length:, 1], color="red")
    ax.set_title("First Linear Layer output")

    # Sigmoid output plot
    ax = fg.add_subplot(1, 4, 3)
    ax.scatter(Z[:data_length, 0], Z[:data_length, 1], color="blue")
    ax.scatter(Z[data_length:, 0], Z[data_length:, 1], color="red")
    ax.set_title("Sigmoid Output")

    # Softmax output
    ax = fg.add_subplot(1, 4, 4)
    ax.scatter(Z2[:data_length, 0], Z2[:data_length, 1], color="blue")
    ax.scatter(Z2[data_length:, 0], Z2[data_length:, 1], color="red")
    ax.set_title("Last Layer Softmax Output")

    plt.show()


def generate_circular_multimodal(modes, r, n_per_mode, var):
    cos_mu = np.cos(np.linspace(0, 2 * np.pi, modes + 1)) * r
    cos_mu = cos_mu[:-1]
    sin_mu = np.sin(np.linspace(0, 2 * np.pi, modes + 1)) * r
    sin_mu = sin_mu[:-1]
    data = np.empty((1, 2))
    for c, s in zip(cos_mu, sin_mu):
        points = mv([c, s], np.asarray([var, 0, 0, var]).reshape((2, 2)), n_per_mode)
        data = np.concatenate((data, points), axis=0)
    return np.asarray(data[1:]), cos_mu, sin_mu

def gaussian_process():
    modes = 11
    n_per_mode = 50
    var = 5
    r = 20
    data, cos_mu, sin_mu = generate_circular_multimodal(modes=modes, r=r, n_per_mode=n_per_mode, var=var)
    fg, ax = plt.subplots(1, 1)
    for i in range(0, data.shape[0], n_per_mode):
        ax.scatter(data[i:i + n_per_mode, 0], data[i:i + n_per_mode, 1])
    ax.scatter(cos_mu, sin_mu, color="red")


    def quadratic_kernel(X, l):
        X1 = np.roll(X, 1, axis=1)
        dist = X.T.dot(X1)
        # print(X[0], X1[0])
        # print(dist.shape)
        sqd = np.power(dist, 2) / l
        sqd = np.asarray(sqd, dtype=np.float128)
        # print(sqd)
        kernel = np.exp(-sqd)
        # print(kernel.shape)
        return kernel


    def vectorized_RBF_kernel(X, sigma):
        # % This is equivalent to computing the kernel on every pair of examples
        X2 = np.sum(np.multiply(X, X), 1)  # sum colums of the matrix
        K0 = X2 + X2.T - 2 * X * X.T
        K = np.power(np.exp(-1.0 / sigma ** 2), K0)
        return K


    def gaussian_process(X, tau, kernel, **kwargs):
        I = np.identity(X.shape[1])
        # print(I.shape)
        Y = mv(np.zeros(X.shape[1]), kernel(X, **kwargs))
        # print(F[0])
        T = mv(Y, I / tau)
        return T


    def linear(X):
        a = np.arange(X.shape[1])[np.newaxis, :]
        a = X + a
        return a

    modes = 1
    r = 10
    n_per_mode = 5
    var = 1
    # X, cos_mu, sin_mu = generate_circular_multimodal(modes=modes, r=r, n_per_mode=n_per_mode, var=var)
    x = np.linspace(0, 100, 200)
    X = np.sin(x) * 15
    X = np.random.normal(X, var, (1, 200))
    data = X
    # data = linear(X)
    for y in X:
        plt.plot(x, data[0])
    Y = gaussian_process(data, .01, vectorized_RBF_kernel, sigma=2)
    kernel = quadratic_kernel(data, 5)
    # print(Y.shape)
    Y = np.asarray(Y, dtype=np.float64)
    plt.figure()
    plt.imshow(np.asarray(kernel, dtype=np.float64))
    plt.colorbar()
    plt.figure()
    plt.plot(x, Y)

def quadratic_manifold_sampling():
    import numpy as np

    n = 1000
    distr = "uniform"

    def sample(distr="uniform", *args):
        if distr == "uniform":
            return np.random.uniform(*args)
        if distr == "normal":
            return np.random.normal(*args)

    theta = np.linspace(0, 2 * np.pi, 360)
    x_axs, y_axs = np.cos(theta), np.sin(theta)
    r = sample(distr, 0, 1, n)
    theta = sample(distr, 0, 2 * np.pi, n)
    x = np.cos(theta) * r
    y = np.sin(theta) * r

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y)
    plt.plot(x_axs, y_axs, color="red")
    plt.show()

    # Quadratic manifold sampling
    x = np.linspace(-1, 1, 100)
    y = np.square(x) - 1

def mixture_model():
    #MIXTURE MODEL
    from scipy.stats import norm, multivariate_normal
    n_samples = 100000
    input_dim = 1
    output_dim = 1
    hidden_size = 4
    activation = "relu"

    p = 0.6
    mu1 = 5
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

    # multi_norm = multivariate_normal.pdf((x, x), np.asarray(mu1*p, 0, 0, mu2*(1-p)).reshape(2,2), np.asarray(sd1*np.sqrt(p), 0, 0, sd2*np.sqrt(1-p)).reshape(2,2))
    # RANDOM SAMPLE
    mode_choice = np.random.uniform(0, 1, n_samples) > p
    mode_choice = np.vstack((1 - mode_choice, mode_choice)).T
    # print(mode_choice.shape, mode_choice[:3])
    X = np.vstack((np.random.normal(mu1, sd1, n_samples), np.random.normal(mu2, sd2, n_samples))).T
    X = (X * mode_choice).flatten()
    X = X[X != 0]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.hist(X, bins=bins, range=(lower, upper), density=True)
    ax.plot(x, normal_mix(p))
    plt.show()

def multivariate_mixture_sampling():
    #MULTIVARIATE MIX DENSITY
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
    x, y = np.meshgrid(x,x)
    pos = np.dstack((x,y))
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