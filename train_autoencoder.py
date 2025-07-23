from autoencoder import autoencoder
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

# parameters definition
my_epochs = 30
input_length = 10

# import data
# prepare input data
x_train = np.zeros((10,10000)) #ça me parait plus propre
x_model = np.array([0,1,2,3,4,5,6,7,8,9])
factor = 100  
for i in range(10000) : 
  random_nb = np.random.random() # génère un flotant aléatoire entre 0 et 1
  bias = np.random.random()*factor #je pense que faire un truc qu'affine plutôt que linéaire peut être interessant
  x_train[:,i] = x_model*random_nb*factor + bias

x_test = np.zeros((10,10000)) #ça me parait plus propre
for i in range(10000) : 
  random_nb = np.random.random() # génère un flotant aléatoire entre 0 et 1
  bias = np.random.random()*factor #je pense que faire un truc qu'affine plutôt que linéaire peut être interessant
  x_test[:,i] = x_model*random_nb*factor + bias

x_train = np.transpose(x_train)
x_test = np.transpose(x_test)

# centrée + normalisé 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# train of the autoencoder from autoencoder.py
autoencoder = autoencoder(my_epochs,input_length)
autoencoder.compile()
autoencoder.model.fit(x_train, x_train, epochs=my_epochs,batch_size=128, shuffle=False, validation_data=(x_test, x_test))

# save parameters of the autoencoder in autoencoder.pth (PyTorch)
# Use of .h5 because we use Keras and not PyTorch
# It's better to have autoencoer.keras but for now it doesn't work with a relative path
autoencoder.save()