import numpy as np
class NeuralNet:
    layers = None #stores the size of each layer

    train_data = None
    train_label = None

    learn = 0

    w = [] #array of w for every layer
    b = []
    z = []
    a = []

    def __init__(self,layers,train_data,train_label,learn):
        self.layers = layers
        self.train_data = train_data
        self.train_label = train_label
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
            print(self.cost())



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
            self.a[layer_num] = self.sigmoid(self.z[layer_num]) #dimensions output by m
        # print("z")
        # print(self.z[layer_num])
        # print("a")
        # print(self.a[layer_num])



    def relu(self,z):
        z[z<0] = 0
        return z

    def cost(self):
        temp1 = (1-self.train_label)*np.log(1-self.a[len(self.a)-1])
        temp2 = (self.train_label)*np.log(self.a[len(self.a)-1])
        return np.sum(temp1+temp2)/self.train_data.shape[1]

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

    def sigmoid(self, arr):
        arr = arr.astype(np.longdouble)
        temp =  1/(1+np.exp(-1*arr))
        temp[temp==0] = 0.000000000000000000000000000000001

        return temp