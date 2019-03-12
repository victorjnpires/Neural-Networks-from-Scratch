# Neural Networks from Scratch functions
# Victor Jose Novaes Pires
# 2019-03-07

__version__ = 1.0

import numpy as np

def get_max_val(shape):
    return np.sqrt(6)/np.sqrt(np.sum(shape))


def random_initializer(shape, max_val=-1, seed=42):
    if max_val <= 0:
        max_val = get_max_val(shape)
    np.random.seed(seed)
    return np.random.rand(*shape)*2*max_val - max_val


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def sigmoid_gradient(z):
    return sigmoid(z)*(1 - sigmoid(z))


def feedforward(X, y, Θ1, Θ2):
    # Input layer activation (with bias)
    a1 = np.hstack((np.ones((X.shape[0], 1)), X))

    # First hidden layer activity
    z2 = Θ1.dot(a1.T)

    # First hidden layer activation
    a2 = sigmoid(z2)

    # First hidden layer activation (with bias)
    a2 = np.vstack((np.ones(a2.shape[1]), a2))

    # Second hidden layer activity
    z3 = Θ2.dot(a2)

    # Second hidden layer activation
    a3 = sigmoid(z3)

    # return a3
    return a3, a2, z2, a1


def cost(X, y, Θ1, Θ2, λ):
    """Computes the cost for neural networks  with regularization

    Args:
        X:  Input
        y:  Labels
        Θ1: First hidden layer
        Θ2: Second hidden layer
        λ:  Regularization

    Returns:
        J: Regularized cost
    """

    m = y.shape[0]

    h, *_ = feedforward(X, y, Θ1, Θ2)

    # Convert an iterable of indices to one-hot encoded labels
    y_ohe = np.eye(y.shape[0], len(np.unique(y)))[y.reshape(-1)]

    # Unregularized cost matrix
    J = (-np.log(h).dot(y_ohe) - np.log(1 - h).dot(1 - y_ohe))/m

    # Sum of the main diagonal of the unregularized cost matrix
    J = (J * np.eye(J.shape[0])).sum()

    if λ != 0:
        # Regularization
        Θ1_vec = Θ1[:, 1:].reshape(-1)
        Θ2_vec = Θ2[:, 1:].reshape(-1)

        # Regularized cost
        J += + λ/(2*m)*(Θ1_vec.dot(Θ1_vec) + Θ2_vec.dot(Θ2_vec))

    return J


def gradient(X, y, Θ1, Θ2, λ, c=None):
    """Computes the gradient for neural networks with regularization

    Args:
        X:  Input
        y:  Labels
        Θ1: First Hidden Layer
        Θ2: Second Hidden Layer
        λ:  Regularization
        c:  Number of classes

    Returns:
        D1, D2: Regularized gradients
    """

    a3, a2, z2, a1 = feedforward(X, y, Θ1, Θ2)

    # c should be passed to ensure that there will be one column for each class.
    # With batches there is the possibility that len(np.unique(y)) != number of
    # classes, and the probability increases as the batch size decreases.
    if c is None:
        c = len(np.unique(y))

    # Convert an iterable of indices to one-hot encoded labels
    y_ohe = np.eye(y.shape[0], c)[y.reshape(-1)]

    # Error terms
    δ3 = a3.T - y_ohe
    δ2 = (δ3.dot(Θ2[:, 1:])).T*sigmoid_gradient(z2)

    # Gradient accumulator
    Δ1 = δ2.dot(a1)
    Δ2 = a2.dot(δ3).T

    # Regularized gradients
    m = y.shape[0]
    D1 = (Δ1 + λ*np.hstack((np.zeros((Θ1.shape[0], 1)), Θ1[:, 1:])))/m
    D2 = (Δ2 + λ*np.hstack((np.zeros((Θ2.shape[0], 1)), Θ2[:, 1:])))/m

    return D1, D2


def predict(X, y, Θ1, Θ2):
    h, *_ = feedforward(X, y, Θ1, Θ2)
    predictions = np.argmax(h, axis=0).reshape(-1, 1)
    return predictions


def accuracy_score(y, predictions):
    accuracy = (predictions == y)
    return np.count_nonzero(accuracy)/len(y)


def cost_and_gradients(Θ, X, y, λ, s):
    """Computes the cost and gradient for optimization

    Args:
        Θ: First and second hidden layers
        X: Input
        y: Labels
        λ: Regularization
        s: First and second hidden layers shapes

    Returns:
        J: Cost
        G: Gradients
    """

    Θ1_size = s[0][0] * s[0][1]
    Θ1 = Θ[:Θ1_size].reshape(s[0])
    Θ2 = Θ[Θ1_size:].reshape(s[1])

    J = cost(X, y, Θ1, Θ2, λ)

    D1, D2 = gradient(X, y, Θ1, Θ2, λ)

    # Gradients as row vector for minimization function
    G = np.concatenate([D1.reshape(-1), D2.reshape(-1)])

    return J, G