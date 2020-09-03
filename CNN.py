from Layer import Layer
import numpy as np
import math

class CNN(Layer):
    def pad(self,data,pad):
        return np.pad(data,((0,0),(pad,pad),(pad,pad),(0,0)),mode = "constant", constant_values=(0,0))



    def forward_prop(self,a):
        z = np.zeros((a.shape[0],a.shape[1]-2,a.shape[2]-2,a.shape[3]))
        for i in range(a.shape[0]): #iterate over num training
            for f in range(a.shape[3]):  # iterate over filters
                for r in range(z.shape[1]):
                    for c in range(z.shape[2]):
                        step_a = a[i,r:r+self.w.shape[0],c:c+self.w.shape[0],:]
                        z[i,r,c,f]=np.sum(np.multiply(step_a, self.w[:,:,:,f])) + self.b[:,:,:,f]
        self.a = self.relu(z)
        return self.a

    def backward_prop(self,prev_da,prev_a,m,iteration):
        dz = self.relu_derivative(prev_da)
        da = np.zeros(prev_a.shape)
        dw = np.zeros(self.w.shape)
        db = np.zeros((1,1,1,self.w.shape[3]))
        for i in range(m):
            for r in range(prev_a.shape[1]-2):
                for c in range(prev_a.shape[2]-2):
                    for f in range(dz.shape[3]):
                        da[i,r:r+self.w.shape[0],c:c+self.w.shape[0],:] += self.w[:][:][:][f] * dz[i][r][c][f]

                        temp_a = prev_a[i,r:r+self.w.shape[0],c:c+self.w.shape[0],:]
                        dw[:,:,:,f]+=temp_a*dz[i][r][c][f]
                        db[:,:,:,f]+=dz[i][r][c][f]
        self.w = self.w - self.learn * dw
        self.b = self.b - self.learn * db
        return da


    def pool(self,a,size,stride):
        z = np.zeros((a.shape[0],math.floor((a.shape[1]-size)/stride)+1,math.floor((a.shape[2]-size)/stride)+1,a.shape[3]))
        for i in range(z.shape[0]): ##iterate over num training
            for f in range(z.shape[3]):
                for r in range(z.shape[1]):
                    for c in range(z.shape[2]):
                        step_a = a[i][r*stride:r*stride+stride][c*stride:c*stride+stride][f]
                        z[i][r][c][f] = np.max(step_a)
        return z


    def relu(self,z):
        rel = np.copy(z)
        rel[rel < 0] = 0
        return rel

    def relu_derivative(self,z):
        der = np.copy(z)
        der[der > 0] = 1
        der[der <= 0] = 0
        return der




