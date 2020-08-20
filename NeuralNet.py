import numpy as np
class NeuralNet:
    input_layer = None
    hidden_layer = None
    output_layer = None
    train_data = None
    train_label = None
    learn = 0
    w1 = None
    b1 = None
    w2 = None
    b2 = None

    def __init__(self,input_layer,hidden_layer, output_layer,train_data,train_label,learn):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.train_data = train_data
        self.train_label = train_label
        self.learn = learn

        #w1 is hidden_layer by input_layer
        w1 = np.random.rand(hidden_layer,input_layer)
        w2 = np.random.rand(output_layer,hidden_layer)

        #b1 is hidden_layer by 1
        b1 = np.random.rand(hidden_layer,1)
        b2 = np.random.rand(output_layer,1)

    def gradient_descent(self,epochs):
        for i in range(epochs):
            a = self.forward_prop()
            self.back_prop(a)

    def forward_prop(self):
        #hidden by input times input by m is hidden by m
        z1 = np.dot(self.w1,self.train_data) + self.b1
        a1 = np.tanh(z1) #dimensions are hidden by m

        #output by hidden times hidden by m is output by m
        z2 = np.dot(self.w2,a1)+self.b2
        a2 = self.sigmoid(z2)
        return a2 #dimensions are output by m

    def back_prop(self,a,z1):
        #output by m - output by 1 is output by m
        dz2 = a-self.train_label
        #output by m times m by output is output by output
        dw2 = np.dot(dz2,np.transpose(a))/self.train_data.shape[1]
        #1 by output
        db2 = np.sum(dz2,axis=1,keepdims=True)/self.train_data.shape[1]  #sums columns

        #hidden by output times output by m is hidden by m
        dz1 = np.multiply(np.dot(np.transpose(self.w2), dz2),1-np.multiply(z1,z1))
        #hidden by m times m by input is hidden by input
        dw1 = np.dot(dz1,np.transpose(self.train_data))/self.train_data.shape[1]
        #dimensions are 1 by input
        db1 = np.sum(dz1,axis = 1, keepdims=True)

        self.w1 = self.w1 - dw1*self.learn
        self.w2 = self.w2 - dw2*self.learn
        self.b1 = self.b1 - db1*self.learn
        self.b2 = self.b2 - db2*self.learn

    def sigmoid(self, arr):
        return 1/(1+np.exp(-1*arr))