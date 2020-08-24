import csv
import numpy as np
from NeuralNet import NeuralNet

data = []
labels = []
with open("/Users/angela/Downloads/heart.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        cur_row = []
        for item in row:
            cur_row.append(float(row.get(item)))
        labels.append(cur_row[13])
        cur_row.pop()
        data.append(cur_row)

data = np.array(data)
data = data.transpose()

#normalization
mean = np.expand_dims(np.sum(data,axis=1)/data.shape[1],axis=1)
data = np.subtract(data,mean)

sigma = np.expand_dims(np.std(data,axis=1),axis=1)
data = np.divide(data,sigma)


labels = np.array(labels)
labels = np.expand_dims(labels,axis=0)
net =  NeuralNet(13,500,1,data,labels,0.2)

net.gradient_descent(100)
