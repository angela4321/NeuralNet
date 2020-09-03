import csv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from NeuralNet import NeuralNet
import os
import random
from Relu import Relu
from Logistic import Logistic
from CNN import CNN



print("here")
files = []
labels = []
path = "/Users/angela/Downloads/kagglecatsanddogs_3367a/PetImages/Cat"
i = 0
for f in os.listdir(path):
    i+=1
    if i>50:
        break
    if os.path.splitext(f)[1] == ".jpg":
        try:
            im = np.asarray(Image.open(os.path.join(path,f)).resize((50,50)))
            files.append([im,0])
        except:
            i-=1

path = "/Users/angela/Downloads/kagglecatsanddogs_3367a/PetImages/Dog"
for f in os.listdir(path):
    i+=1
    if i>100:
        break
    if os.path.splitext(f)[1] == ".jpg":
        try:
            im = np.asarray(Image.open(os.path.join(path, f)).resize((50,50)))
            files.append([im, 1])
        except ValueError:
            i-=1
            print("here")

random.shuffle(files)
data =[]
for i in files:
    data.append(i[0])
    labels.append(i[1])

data = np.array(data)
data = data/100
#data = np.swapaxes(data,0,3)
#
# #normalization
# mean = np.expand_dims(np.sum(data,axis=1)/data.shape[1],axis=1)
# data = np.subtract(data,mean)
#
# sigma = np.expand_dims(np.std(data,axis=1),axis=1)
# data = np.divide(data,sigma)
#


data_train = data[:70,:,:,:]
data_val = data[70:,:,:,:]
labels = np.expand_dims(labels,axis=0)
labels_train = labels[:,:70]
labels_val = labels[:,70:]

layers = [0,-1,50,1]
layer_type = [CNN(), CNN(),Relu(),Logistic()]
net =  NeuralNet(layers,layer_type,data_train,labels_train,data_val,labels_val,0.000005,data_train.shape[0],batch = 16)

net.gradient_descent(10)

print(net.c)
print(net.v)
plt.plot(net.c,label = "training")
plt.plot(net.v,label = "validation")
plt.legend()
plt.show()