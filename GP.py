import distributions as distr


def gaussian_process_model():
    modes = 11
    n_per_mode = 50
    var = 5
    r = 20
    data, cos_mu, sin_mu = distr.generate_circular_multimodal(modes=modes, r=r, n_per_mode=n_per_mode, var=var)
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
    # X, cos_mu, sin_mu = distr.generate_circular_multimodal(modes=modes, r=r, n_per_mode=n_per_mode, var=var)
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
