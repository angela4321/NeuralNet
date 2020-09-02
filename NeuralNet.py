import numpy as np
from scipy import special
from numpy import inf
import math
class NeuralNet:

    def __init__(self,layers,layer_type,train_data,train_label,val_data,val_label,learn,b1=0.9,b2 = 0.99,batch = 0):
        self.layers = layers
        self.layer_type = layer_type
        self.train_data = train_data
        self.train_label = train_label
        self.val_data = val_data
        self.val_label = val_label
        self.learn = learn
        self.b1=b1
        self.b2=b2
        self.batch = batch
        if batch==0:
            self.batch = train_data.shape[1]

        self.w = []
        self.b = []
        self.z = []
        self.a = []

        self.c = []
        self.v = []

        self.vdw = []
        self.vdb = []

        self.sdw = []
        self.sdb = []


        self.w.append(None)
        self.b.append(None)
        self.z.append(None)
        self.a.append(None)
        self.vdw.append(None)
        self.vdb.append(None)
        self.sdw.append(None)
        self.sdb.append(None)
        for i in range(len(layers)-1):
            self.w.append(np.random.rand(layers[i+1],layers[i]).astype(np.longdouble)/10)
            self.b.append(np.zeros((layers[i+1],1)).astype(np.longdouble))
            self.z.append(None)
            self.a.append(None)
            self.vdw.append(np.zeros((layers[i+1],layers[i])))
            self.vdb.append(np.zeros((layers[i+1],1)))
            self.sdw.append(np.zeros((layers[i+1],layers[i])))
            self.sdb.append(np.zeros((layers[i+1],1)))
        self.a[0] = train_data


    def gradient_descent(self,epochs):
        for i in range(epochs):
            for self.j in range(math.floor(self.train_data.shape[1]/self.batch)):
                #make mini batches
                end = 0
                if (self.j+1)*self.batch>self.train_data.shape[1]:
                    end = self.train_data.shape[1]
                else:
                    end = (self.j+1)*self.batch
                self.a[0] = self.train_data[:,self.j*self.batch:end]

                #forward prop
                for j in range(len(self.layers)-1):
                    self.forward_prop(j+1)

                # back prop
                temp_label = self.train_label[:, self.j * self.batch:self.j * self.batch + self.a[0].shape[1]]
                prev_da = -np.divide(temp_label, self.a[len(self.a) - 1])
                prev_da = prev_da + np.divide((1 - temp_label), (1 - self.a[len(self.a) - 1]))
                for j in range(len(self.layers) - 1):
                    da = self.back_prop(prev_da, len(self.layers) - 1 - j)
                    prev_da = da
                # print("loss: "+str(self.cost()))

            #calculate cost for training
            prev_a = self.train_data
            for j in range(len(self.layers) - 1):
                prev_a = self.predict(prev_a, j + 1)
            self.c.append(self.layer_type[len(self.layer_type)-1].cost(self.train_label, prev_a))

            #calculate cost for validation set
            prev_a = self.val_data
            for j in range(len(self.layers) - 1):
                prev_a = self.predict(prev_a, j + 1)
            self.v.append(self.layer_type[len(self.layer_type)-1].cost(self.val_label, prev_a))

    def forward_prop(self,layer_num):
        self.z[layer_num] = np.dot(self.w[layer_num], self.a[layer_num - 1]) + self.b[layer_num]
        self.a[layer_num] = self.layer_type[layer_num].activation(self.z[layer_num]) #dimensions are next by m

        self.a[layer_num] = np.nan_to_num(self.a[layer_num])
        self.a[layer_num][self.a[layer_num]==0] = 0.0000000000000000001
        self.a[layer_num][self.a[layer_num]==1] = 0.9999999999999999999
        self.z[layer_num][self.z[layer_num] == inf] = 9999999999999999999


    def back_prop(self, da, layer_num):
        dz=np.multiply(da, self.layer_type[layer_num].derivative(self.z[layer_num]))
        dz[dz == inf] = 999999999999999999999999
        dw = np.dot(dz, np.transpose(self.a[layer_num - 1])) / self.train_data.shape[1]

        db = np.sum(dz, axis=1, keepdims=True) / self.train_data.shape[1]

        temp = np.dot(np.transpose(self.w[layer_num]), dz)

        self.adam(layer_num,dw,db)


        return temp

    def adam(self,layer_num,dw,db):
        self.vdw[layer_num] = (self.b1) * self.vdw[layer_num] + (1 - self.b1) * dw
        vdwc = self.vdw[layer_num] / (1 - pow(self.b1, self.j + 1))

        self.vdb[layer_num] = self.b1 * self.vdb[layer_num] + (1 - self.b1) * db
        vdbc = self.vdb[layer_num] / (1 - pow(self.b1, self.j + 1))

        self.sdw[layer_num] = self.b2 * self.sdw[layer_num] + (1 - self.b2) * np.power(dw, 2)
        sdwc = self.sdw[layer_num] / (1 - pow(self.b2, self.j + 1))

        self.sdb[layer_num] = self.b2 * self.sdb[layer_num] + (1 - self.b2) * np.power(db, 2)
        sdbc = self.sdb[layer_num] / (1 - pow(self.b2, self.j + 1))

        # update weights
        self.w[layer_num] = self.w[layer_num] - self.learn * vdwc / (np.sqrt(sdwc) + 10 ** -8)
        self.b[layer_num] = self.b[layer_num] - self.learn * vdbc / (np.sqrt(sdbc) + 10 ** -8)


    def predict(self,a_prev,layer_num):

        z = np.dot(self.w[layer_num], a_prev) + self.b[layer_num]
        a = self.layer_type[layer_num].activation(z)  # dimensions are next by m
        a = np.nan_to_num(a)
        a[a == 0] = 0.0000000000000000001
        a[a == 1] = 0.9999999999999999999
        return a




