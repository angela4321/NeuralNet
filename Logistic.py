from Layer import Layer
from scipy import special
import numpy as np
from math import inf

class Logistic(Layer):
    def cost(self,label,predict):
        print(predict)
        temp1 = np.multiply((1 - label), np.log(1 - predict))
        temp2 = np.multiply((label), np.log(predict))
        return -1 * np.sum(temp1 + temp2) / label.shape[1]

        # pred = np.copy(predict)
        # pred[pred>0.5]=1
        # pred[pred<=0.5] = 0
        # return np.sum(np.abs(label-pred))

    def forward_prop(self, prev_a):
        self.z = np.dot(self.w, prev_a) + self.b
        self.a = self.activation(self.z)  # dimensions are next by m
        self.a = np.nan_to_num(self.a)
        # self.a[self.a == 0] = 0.0000000000000000001
        # self.a[self.a == 1] = 0.9999999999999999999
        # self.z[self.z == inf] = 9999999999999999999
        return self.a

    def backward_prop(self, da,prev_a,m,iteration):
        dz = np.multiply(da, self.derivative(self.z))
        #dz[dz == inf] = 999999999999999999999999
        dw = np.dot(dz, np.transpose( prev_a)) / m

        db = np.sum(dz, axis=1, keepdims=True) / m

        temp = np.dot(np.transpose(self.w), dz)

        self.adam(dw, db,iteration)

        return temp


    def activation(self,z):
        return special.expit(z)

    def derivative(self,z):
        temp = special.expit(z)
        return temp * (1 - temp)

