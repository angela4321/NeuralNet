import numpy as np
from CNN import CNN
import math
class NeuralNet:

    def __init__(self,layers,layer_type,train_data,train_label,val_data,val_label,learn,m,b1=0.9,b2 = 0.99,batch = 0):
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
        self.m=m #num training examples
        if batch==0:
            self.batch =m


        self.c = []
        self.v = []
        # [0, -1, -2, 50, 1]

        for i in range(len(layers)-1):
            if layers[i]==-1 and layers[i+1]!=-1: # flatten the CNN
                layers[i] = (train_data.shape[0])*(train_data.shape[1]-2)*(train_data.shape[2]-2)

            if layers[i+1]==-1: #CNN layer
                w = np.random.rand(3,3,3,3).astype(np.longdouble)/10000
                b = np.zeros((1,1,1,3))
                self.layer_type[i+1].initialize_vars(w=w,b=b)
            else:
                w = np.random.rand(layers[i + 1], layers[i]).astype(np.longdouble) / 10
                b = np.zeros((layers[i + 1], 1)).astype(np.longdouble)
                z = None
                a = None
                vdw = np.zeros((layers[i+1],layers[i]))
                vdb = np.zeros((layers[i+1],1))
                sdw = np.zeros((layers[i+1],layers[i]))
                sdb = np.zeros((layers[i+1],1))
                self.layer_type[i+1].initialize_vars(w,b,z,a,vdw,vdb,sdw,sdb)
        self.layer_type[0].initialize_vars(train_data)


    def gradient_descent(self,epochs):
        for i in range(epochs):
            for self.j in range(math.floor(self.m/self.batch)):
                #make mini batches
                end = 0
                if (self.j+1)*self.batch>self.m:
                    end = self.m
                else:
                    end = (self.j+1)*self.batch
                self.layer_type[0].a = self.train_data[:,self.j*self.batch:end]
                if type(self.layer_type[0])==CNN:

                    self.layer_type[0].a = self.train_data[:,:,:,self.j*self.batch:end]

                #forward prop
                prev_a = self.layer_type[0].a
                for j in range(len(self.layers)-1):
                    if type(self.layer_type[j])==CNN and type(self.layer_type[j+1])!=CNN:
                        prev_a = prev_a.reshape(-1,prev_a.shape[-1])
                    prev_a = self.layer_type[j+1].forward_prop(prev_a)
                    # self.forward_prop(j+1)

                # back prop
                temp_label = self.train_label[:, self.j * self.batch:self.j * self.batch + prev_a.shape[1]]
                prev_da = -np.divide(temp_label, prev_a)
                prev_a[prev_a==1]=0.99999999999
                prev_da = prev_da + np.divide((1 - temp_label), (1 - prev_a))

                for j in range(len(self.layers) - 1):
                    prev_a = self.layer_type[len(self.layers)-1-j-1].a
                    if type(self.layer_type[len(self.layers)-1-j-1])==CNN and type(self.layer_type[len(self.layers)-1-j])!=CNN:
                        prev_a = prev_a.reshape(-1,prev_a.shape[-1])

                    prev_da = self.layer_type[len(self.layers)-1-j].backward_prop(prev_da,prev_a,self.m,self.j) #da, preva

                    if type(self.layer_type[len(self.layers) - 1 - j - 1]) == CNN and type(
                            self.layer_type[len(self.layers) - 1 - j]) != CNN:
                        prev_da = prev_da.reshape(self.train_data.shape[0], self.train_data.shape[1] - 2,
                                                  self.train_data.shape[2] - 2, -1)

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

    def predict(self,a_prev,layer_num):

        z = np.dot(self.layer_type[layer_num].w, a_prev) + self.layer_type[layer_num].b
        a = self.layer_type[layer_num].activation(z)  # dimensions are next by m
        a = np.nan_to_num(a)
        a[a == 0] = 0.0000000000000000001
        a[a == 1] = 0.9999999999999999999
        return a




