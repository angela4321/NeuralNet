import csv
import numpy as np
from matplotlib import pyplot as plt
from NeuralNet import NeuralNet
from Relu import Relu
from Logistic import Logistic

data = []
labels = []
with open("/Users/angela/Downloads/heart.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        cur_row = []
        for item in row:
            cur_row.append(float(row.get(item)))
        data.append(cur_row)


data = np.array(data)
np.random.shuffle(data)
data = data.transpose()

labels = data[13:]
data = data[0:13,]

#normalization
mean = np.expand_dims(np.sum(data,axis=1)/data.shape[1],axis=1)
data = np.subtract(data,mean)

sigma = np.expand_dims(np.std(data,axis=1),axis=1)
data = np.divide(data,sigma)



data_train = data[:,:230]
data_val = data[:,230:]

labels_train = labels[:,:230]
labels_val = labels[:,230:]

layers = [13,75,50,20,1]
layer_type = [Relu(),Relu(),Relu(),Relu(),Logistic()]
net =  NeuralNet(layers,layer_type,data_train,labels_train,data_val,labels_val,0.0005,data_train.shape[1],batch = 16)

net.gradient_descent(100)

plt.plot(net.c,label = "training")
plt.plot(net.v,label = "validation")
plt.legend()
plt.show()