from Layer import Layer
import numpy as np


class Relu(Layer):

    def activation(self, z):
        rel = np.copy(z)
        rel[rel < 0] = 0
        return rel

    def derivative(self, z):
        der = np.copy(z)
        der[der > 0] = 1
        der[der <= 0] = 0
        return der