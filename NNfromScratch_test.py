# Neural Networks from Scratch functions unittest
# Victor Jose Novaes Pires
# 2019-03-15

import unittest
import numpy as np
import NNfromScratch as nnfs

z = np.linspace(-10, 10, int(1e6))

shape = [int(1e3), int(1e3)]

num_elements = 5_000
num_features = 100
hidden_layer_size = 32
classes = 7
X = np.random.rand(num_elements, num_features)
y = np.random.randint(low=0, high=classes, size=[num_elements, 1])
Θ1 = nnfs.random_initializer([hidden_layer_size, (num_features + 1)])
Θ2 = nnfs.random_initializer([classes, (hidden_layer_size + 1)])
λ = 1 # Regularization

class TestNNfromScratch(unittest.TestCase):
    def test_sigmoid(self):
        np.testing.assert_equal(
                nnfs.sigmoid(z),
                1/(1 + np.exp(-1*z))
        )


    def test_sigmoid_gradient(self):
        np.testing.assert_equal(
                nnfs.sigmoid_gradient(z),
                1/(1 + np.exp(-1*z)) * (1 - 1/(1 + np.exp(-1*z)))
        )


    def test_get_max_val(self):
        self.assertEqual(nnfs.get_max_val(shape),
                         np.sqrt(6)/np.sqrt(np.sum(shape)))


    def test_random_initializer(self):
        matrix = nnfs.random_initializer(shape)
        max_val = np.sqrt(6)/np.sqrt(np.sum(shape))
        self.assertGreaterEqual(matrix.min(), -max_val)
        self.assertLessEqual(matrix.max(), max_val)

        array = np.array([0, 1])*2*max_val - max_val
        self.assertEqual(array.min(), -max_val)
        self.assertEqual(array.max(), max_val)


    def test_feedforward(self):
        a3, a2, z2, a1 = nnfs.feedforward(X, Θ1, Θ2)
        self.assertEqual(a1.shape, (X.shape[0], (X.shape[1] + 1)))
        self.assertEqual(z2.shape, (Θ1.shape[0], X.shape[0]))
        self.assertEqual(a2.shape, ((Θ1.shape[0] + 1), X.shape[0]))
        self.assertEqual(a3.shape, (Θ2.shape[0], X.shape[0]))


    def test_cost(self):
        J = nnfs.cost(X, y, Θ1, Θ2, λ)
        self.assertEqual(type(J), np.float64)


    def test_gradient(self):
        D1_1, D2_1 = nnfs.gradient(X, y, Θ1, Θ2, λ)
        self.assertEqual(D1_1.shape, Θ1.shape)
        self.assertEqual(D2_1.shape, Θ2.shape)
        D1_2, D2_2 = nnfs.gradient(X, y, Θ1, Θ2, λ, c=classes)
        np.testing.assert_equal(D1_1, D1_2)
        np.testing.assert_equal(D2_1, D2_2)


    def test_predict(self):
        predictions = nnfs.predict(X, y, Θ1, Θ2)
        self.assertEqual(predictions.shape, y.shape)


    def test_accuracy_score(self):
        predictions = nnfs.predict(X, y, Θ1, Θ2)
        self.assertEqual(
                nnfs.accuracy_score(y, predictions),
                np.sum(predictions == y)/len(y)
        )


    def test_cost_and_gradients(self):
        j = nnfs.cost(X, y, Θ1, Θ2, λ)
        D1, D2 = nnfs.gradient(X, y, Θ1, Θ2, λ)
        g = np.concatenate([D1.reshape(-1), D2.reshape(-1)])
        J, G = nnfs.cost_and_gradients(
                np.concatenate([Θ1.reshape(-1), Θ2.reshape(-1)]),
                X, y, λ, (Θ1.shape, Θ2.shape)
        )
        np.testing.assert_equal(J, j)
        np.testing.assert_equal(G, g)


if __name__ == '__main__':
    unittest.main()