from Layer import Layer
import numpy as np
from scipy import special

class Logistic(Layer):
    def cost(self,label,predict):
        temp1 = np.multiply((1 - label), np.log(1 - predict))
        temp2 = np.multiply((label), np.log(predict))
        return -1 * np.sum(temp1 + temp2) / label.shape[1]

    def forward_prop(self, prev_a):
        self.z = np.dot(self.w, prev_a) + self.b
        self.a = self.activation(self.z)  # dimensions are next by m
        return self.a

    def backward_prop(self, da,prev_a,train_data,iteration):
        dz = np.multiply(da, self.derivative(self.z))
        #dz[dz == inf] = 999999999999999999999999
        dw = np.dot(dz, np.transpose( prev_a)) / train_data.shape[1]

        db = np.sum(dz, axis=1, keepdims=True) / train_data.shape[1]

        temp = np.dot(np.transpose(self.w), dz)

        self.adam(dw, db,iteration)

        return temp


    def activation(self,z):
        return special.expit(z)

    def derivative(self,z):
        temp = special.expit(z)
        return temp * (1 - temp)

