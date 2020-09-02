from Layer import Layer
import numpy as np
from scipy import special

class Logistic(Layer):
    def cost(self,label,predict):
        temp1 = np.multiply((1 - label), np.log(1 - predict))
        temp2 = np.multiply((label), np.log(predict))
        return -1 * np.sum(temp1 + temp2) / label.shape[1]

    def activation(self,z):
        return special.expit(z)

    def derivative(self,z):
        temp = special.expit(z)
        return temp * (1 - temp)

