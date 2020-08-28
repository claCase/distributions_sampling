import matplotlib.pyplot as plt
import numpy as np

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
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    z = multivariate_normal(mu, cov)

    x2, y2 = np.meshgrid(np.linspace(mu2 - 3, mu2 + 3, 50), np.linspace(mu2 - 3, mu2 + 3, 50))
    mu2 = np.asarray((mu2, mu2))
    cov2 = np.asarray((var2, 0, 0, var2)).reshape((2, 2))
    pos2 = np.empty(x2.shape + (2,))
    pos2[:, :, 0] = x2
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

