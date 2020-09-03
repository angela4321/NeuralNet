from Layer import Layer
import numpy as np
import math

class CNN(Layer):
    def pad(self,data,pad):
        return np.pad(data,((0,0),(pad,pad),(pad,pad),(0,0)),mode = "constant", constant_values=(0,0))

    def conv(self,a,w,b):
        z = np.zeros((a.shape[0],a.shape[1]-2,a.shape[2]-2,a.shape[3]))
        for i in range(a.shape[0]): #iterate over num training
            for f in range(a.shape[3]):  # iterate over filters
                for r in range(z.shape[1]):
                    for c in range(z.shape[2]):
                        step_a = a[i][r:r+w.shape[0]][c:c+w.shape[0]]
                        z[i][r][c][f]=np.sum(np.multiply(step_a, w[:][:][:][f])) + b[:][:][:][f]
        return z

    def back_prop(self,dz,a,w):
        da = np.zeros(a.shape)
        dw = np.zeros(w.shape)
        db = np.zeros((1,1,1,w.shape[3]))
        for i in range(a.shape[0]):
            for r in range(a.shape[1]):
                for c in range(a.shape[2]):
                    for f in range(dz.shape[3]):
                        da[r:r+w.shape[0]][c:c+w.shape[0]]+= w[:][:][:][f] * dz[i][r][c][f]

                        temp_a = a[i][r:r+w.shape[0]][c:c+w.shape[0]]
                        dw[:][:][:][c]+=temp_a*dz[i][r][c][f]
                        db[:][:][:][c]+=dz[i][r][c][f]

        return da,dw,db


    def pool(self,a,size,stride):
        z = np.zeros((a.shape[0],math.floor((a.shape[1]-size)/stride)+1,math.floor((a.shape[2]-size)/stride)+1,a.shape[3]))
        for i in range(z.shape[0]): ##iterate over num training
            for f in range(z.shape[3]):
                for r in range(z.shape[1]):
                    for c in range(z.shape[2]):
                        step_a = a[i][r*stride:r*stride+stride][c*stride:c*stride+stride][f]
                        z[i][r][c][f] = np.max(step_a)
        return z







