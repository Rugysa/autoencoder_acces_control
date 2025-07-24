from autoencoder import autoencoder
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

# Parameters definition
my_epochs = 30
input_length = 10


# Import data
# Here : prepare input data
# Input data is 10 000 vectors (with a length of 10) of number link by an affine relationship 

# Base for affine relationship
x_model = np.array([0,1,2,3,4,5,6,7,8,9])
factor = 100  

# Create x_train, trainning data
x_train = np.zeros((10,10000)) 
for i in range(10000) : 
  random_nb = np.random.random() # float between 0 and 1
  bias = np.random.random()*factor
  x_train[:,i] = x_model*random_nb*factor + bias

# Create x_test, validation data
x_test = np.zeros((10,10000))
for i in range(10000) : 
  random_nb = np.random.random() # float between 0 and 
  bias = np.random.random()*factor
  x_test[:,i] = x_model*random_nb*factor + bias

# Preparation of data for the autoencoder
x_train = np.transpose(x_train)
x_test = np.transpose(x_test)
# Data centred and normalised
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Train of the autoencoder from autoencoder.py
autoencoder = autoencoder(my_epochs,input_length)
autoencoder.compile()
autoencoder.model.fit(x_train, x_train, epochs=my_epochs,batch_size=128, shuffle=False, validation_data=(x_test, x_test))


# Save parameters of the autoencoder in autoencoder.pth (PyTorch)
# Here : use of .keras because we use Keras and not PyTorch
autoencoder.save()