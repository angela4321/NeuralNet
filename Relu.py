from Layer import Layer
import numpy as np
from math import inf


class Relu(Layer):



    def forward_prop(self,prev_a):
        self.z = np.dot(self.w, prev_a) + self.b
        self.a = self.activation(self.z)  # dimensions are next by m
        return self.a
        self.a = np.nan_to_num(self.a)
        self.a[self.a == 0] = 0.0000000000000000001
        self.a[self.a == 1] = 0.9999999999999999999
        self.z[self.z == inf] = 9999999999999999999

    def activation(self,z):
        rel = np.copy(z)
        rel[rel < 0] = 0
        return rel

    def backward_prop(self, da,prev_a,m,iteration):
        dz = np.multiply(da, self.derivative(self.z))
        #dz[dz == inf] = 999999999999999999999999
        dw = np.dot(dz, np.transpose(prev_a)) / m

        db = np.sum(dz, axis=1, keepdims=True) /m

        temp = np.dot(np.transpose(self.w), dz)

        self.adam(dw, db,iteration)

        return temp


    def derivative(self,z):
        der = np.copy(z)
        der[der > 0] = 1
        der[der <= 0] = 0
        return der