#%%
## This is an implementation of some of the ideas described in the paper "Deep neural networks and mixed integer linear optimization", 
## by M. Fischetti and J. Jo (https://doi.org/10.1007/s10601-018-9285-6). 

## First, a neural networt is trained to predic digits using the MNIST set. The code is based on 
## https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d

## Then, an "adversarial example" is generated, i.e., and image is selected from the test set and modified 
## as little as possible while causing its classification to change from que original one to another chosen digit.
## To accomplish that, an MIP problem is solved using Gurobi.


#%%
## Import libraries

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gurobipy import *


#%%
##########################################
## PART 1 - Training the neural network ##
##########################################


#%%
## Set seeds for reproducibility

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(0)
tf.random.set_seed(0)


#%%
## Import data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#%%
## Show a training set image (as an example)

image_index = 1000 # Choose any number up to 60,000
print("Image label:", y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')


#%%
## Reshape and rescale data

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


#%%
## Architecture used by Fischetti and Jo (probably, since a few details are not specified)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation

nn_model = Sequential()
nn_model.add(Flatten())
nn_model.add(Dense(8, activation=tf.nn.relu))
nn_model.add(Dense(8, activation=tf.nn.relu))
nn_model.add(Dense(8, activation=tf.nn.relu))
nn_model.add(Dense(10, activation=tf.nn.relu))
nn_model.add(Activation(tf.nn.softmax))


#%%
## Train model

nn_model.compile(optimizer='adam', # This is not specified in the paper
              loss='sparse_categorical_crossentropy', # This is not specified in the paper
              metrics=['accuracy'])
nn_model.fit(x=x_train,y=y_train, epochs=5)


#%%
## Print model summary

print(nn_model.summary())


#%%
## Evaluate model on test set and print summary

nn_model.evaluate(x_test, y_test)


#%%
##############################################
## PART 2 - Building an adversarial example ##
##############################################


#%%
## Set parameters

K = 4                   # Layers are numbered 0, 1, ..., K, so there are K+1 layers
n = [784, 8, 8, 8, 10]  # n[k] is the number of neurons in layer[k]

#%%
## Get model weights and feed into weights matrices W's and biases b's

weights = nn_model.get_weights()

W = []
W.append(weights[0])
W.append(weights[2])
W.append(weights[4])
W.append(weights[6])

b = []
b.append(weights[1])
b.append(weights[3])
b.append(weights[5])
b.append(weights[7])


#%%
## Read input image

image_index = 0 # Choose any number up to 10,000
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = nn_model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print("Predictions:", pred)
print("Predicted value:", pred.argmax())
print("True value:", y_test[image_index])


#%%
## Create Gurobi MIP model

mip_model = Model()

## Create x variables (continuous)
## Variable x[j,k] is associated with node j of layer k
## For k = 0, x[j,0] is the jth value of the input vector
x = {}
for j in range(n[0]):
    x[j+1,0] = mip_model.addVar(vtype=GRB.CONTINUOUS, ub=1.0, name='x'+'_'+str(j+1)+'_'+str(0))
mip_model.update()

for k in range(1,K+1):
    for j in range(n[k]):
        x[j+1,k] = mip_model.addVar(vtype=GRB.CONTINUOUS, name='x'+'_'+str(j+1)+'_'+str(k))
mip_model.update()


## Create s variables (continuous)
## Variable s[j,k] is associated with node j of layer k
s = {}
for k in range(1,K+1):
    for j in range(n[k]):
        s[j+1,k] = mip_model.addVar(vtype=GRB.CONTINUOUS, name='s'+'_'+str(j+1)+'_'+str(k))
mip_model.update()


## Create z variables (binary)
## Variable z[j,k] is associated with node j of layer k
z = {}
for k in range(1,K+1):
    for j in range(n[k]):
        z[j+1,k] = mip_model.addVar(vtype=GRB.BINARY, name='z'+'_'+str(j+1)+'_'+str(k))
mip_model.update()


## Create d variables (continuous)
## Variable d[j] is associated with variable x[j,0]
d = {}
for j in range(n[0]):
    d[j+1] = mip_model.addVar(vtype=GRB.CONTINUOUS, name='d'+'_'+str(j+1))
mip_model.update()


#%%
## Add affine transoformation constraints

for k in range(1,K+1):
    for j in range(1,n[k]+1):
        mip_model.addConstr(quicksum(W[k-1][i][j-1] * x[i+1,k-1] for i in range(n[k-1])) + b[k-1][j-1] == x[j,k] - s[j,k], name='a'+'_'+str(j)+'_'+str(k))
mip_model.update()


#%%
## Add activation constraints

M = 1000     # Big-M for bounding the constraints

for k in range(1,K+1):
    for j in range(1,n[k]+1):
        mip_model.addConstr(x[j,k] <= M - M * z[j,k], name='b'+'_'+str(j)+'_'+str(k))
        mip_model.addConstr(s[j,k] <= M * z[j,k], name='c'+'_'+str(j)+'_'+str(k))
mip_model.update()


#%%
## Add constraints limiting the variation of the input variables and the values of the image

input_image = x_test[image_index].flatten()

for j in range(n[0]):
    mip_model.addConstr(x[j+1,0] - input_image[j] <= d[j+1])
    mip_model.addConstr(x[j+1,0] - input_image[j] >= -d[j+1])
mip_model.update()


#%%

## Add constraints to force the classification of the image as a given value

predicted_digit = 5 # Choose any digit from 0 to 9

for j in range(10):
    if j != predicted_digit:
        mip_model.addConstr(x[predicted_digit+1,K] >= 1.2 * x[j+1,K])


#%%
## Set objective

mip_model.setObjective(quicksum(d[j+1] for j in range(n[0])), GRB.MINIMIZE)


#%%
## Write model do disk (optional)

#mip_model.write("model.lp")


#%%
## Optimize model

mip_model.optimize()


#%%
## Show solution (optional)

# Show solution on the x[j,k] variables (optional)
#for k in range(K+1):
#    for j in range(n[k]):
#        print("x_"+str(j+1)+"_"+str(k), x[j+1,k])

# Show solution on the s[j,k] variables (optional)
#for k in range(1,K+1):
#    for j in range(n[k]):
#        print("s_"+str(j+1)+"_"+str(k), s[j+1,k])

# Show solution on the z[j,k] variables (optional)
#for k in range(1,K+1):
#    for j in range(n[k]):
#        print("z_"+str(j+1)+"_"+str(k), z[j+1,k])

# Show solution on the d[j] variables (optional)
#for j in range(n[0]):
#    print("d_"+str(j+1), d[j+1])


#%%
# Retrieve modified image from the MIP solution

var_values = []

for k in range(1):
    for j in range(n[k]):
        var_values.append(x[j+1,k].X)

input_image_modified = np.array(var_values)


# %%
## Show modified image with new classification

plt.imshow(input_image_modified.reshape(28, 28),cmap='Greys')
pred = nn_model.predict(input_image_modified.reshape(1, 28, 28, 1))
print("Predictions:", pred)
print("Predicted value:", pred.argmax())


# %%
