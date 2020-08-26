import numpy as np
from scipy import special
from numpy import inf
class NeuralNet:
    layers = None #stores the size of each layer

    train_data = None
    train_label = None

    val_data = None
    val_label = None

    learn = 0

    w = [] #array of w for every layer
    b = []
    z = []
    a = []

    c = []  #training loss
    v = []  #validation loss

    def __init__(self,layers,train_data,train_label,val_data,val_label,learn):
        self.layers = layers
        self.train_data = train_data
        self.train_label = train_label
        self.val_data = val_data
        self.val_label = val_label
        self.learn = learn

        self.w.append(None)
        self.b.append(None)
        self.z.append(None)
        self.a.append(None)
        for i in range(len(layers)-1):
            self.w.append(np.random.rand(layers[i+1],layers[i]).astype(np.longdouble)/100)


            self.b.append(np.random.rand(layers[i+1],1).astype(np.longdouble))
            self.z.append(np.random.rand(1,1))
            self.a.append(np.random.rand(1,1))
        self.a[0] = train_data


    def gradient_descent(self,epochs):
        for i in range(epochs):
            for j in range(len(self.layers)-1):
                self.forward_prop(j+1)
            prev_da = -self.train_label/self.a[len(self.a)-1]
            prev_da = prev_da+(1-self.train_label)/(1-self.a[len(self.a)-1])
            for j in range(len(self.layers)-1):
                da = self.back_prop(prev_da,len(self.layers)-1-j)
                prev_da = da
            # print("loss: "+str(self.cost()))
            self.c.append(self.cost(self.train_label,self.a[len(self.a)-1]))
            prev_a = self.val_data
            for j in range(len(self.layers)-1):
                prev_a = self.val_prediction(prev_a,j+1)
            self.v.append(self.cost(self.val_label,prev_a))


    def val_prediction(self,a_prev,layer_num):
        if layer_num != len(self.layers) - 1:
            # next by current times current by m is next by m
            z = np.dot(self.w[layer_num], a_prev) + self.b[layer_num]
            a = self.relu(z)  # dimensions are next by m

        else:
            # next by current times current by m is next by m
            z= np.dot(self.w[layer_num], a_prev) + self.b[layer_num]
            a = special.expit(z)
            np.nan_to_num(a)
            a[a == 0] = 0.00000000000001
            a[a == 1] = 0.99999999999999
        return a

    def forward_prop(self,layer_num):
        if layer_num!=len(self.layers)-1:
            #next by current times current by m is next by m
            self.z[layer_num] = np.dot(self.w[layer_num],self.a[layer_num-1]) + self.b[layer_num]
            self.a[layer_num] = self.relu(self.z[layer_num]) #dimensions are next by m

            dropout = np.random.rand(self.a[layer_num].shape[0],self.a[layer_num].shape[1])<0.8
            #self.a[layer_num] =  np.multiply(self.a[layer_num],dropout)/0.8
        else:
            #next by current times current by m is next by m
            self.z[layer_num] = np.dot(self.w[layer_num],self.a[layer_num-1])+self.b[layer_num]
            #self.a[layer_num] = self.sigmoid(self.z[layer_num]) #dimensions output by m
            self.a[layer_num] = special.expit(self.z[layer_num])
            np.nan_to_num(self.a[layer_num])
            self.a[layer_num][self.a[layer_num]==0] = 0.00000000000001
            self.a[layer_num][self.a[layer_num]==1] = 0.99999999999999
        self.z[layer_num][self.z[layer_num] == inf] = 0.99999999999999
        # print("z")
        # print(self.z[layer_num])
        # print("a")
        # print(self.a[layer_num])



    def relu(self,z):
        z[z<0] = 0
        return z

    def cost(self,label, predict):
        temp1 = (1-label)*np.log(1-predict)
        temp2 = (label)*np.log(predict)
        return -1*np.sum(temp1+temp2)/label.shape[1]

    def relu_derivative(self,z):
        z[z>=0] = 1
        z[z<0] = 0
        return z

    def sigmoid_derivative(self,z):
        return z*(1-z)

    def back_prop(self,da,layer_num):
        dz = None
        if(layer_num==len(self.layers)-1):
            dz = np.multiply(da,self.sigmoid_derivative(self.z[layer_num]))

        else:
            dz = np.multiply(da,self.relu_derivative(self.z[layer_num]))

        dz[dz==inf] = 0.99999999999999
        dw = np.dot(dz,np.transpose(self.a[layer_num-1]))/self.train_data.shape[1]
        db = np.sum(dz,axis=1,keepdims=True)/self.train_data.shape[1]


        self.w[layer_num] = self.w[layer_num] - dw*self.learn
        self.b[layer_num] = self.b[layer_num] - db*self.learn
        # print("dz")
        # print(dz)
        # print("dw")
        # print(dw)
        # print("db")
        # print(db)
        return np.dot(np.transpose(self.w[layer_num]),dz)