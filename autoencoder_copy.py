import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
import numpy as np
from sklearn.preprocessing import StandardScaler

# Single fully-connected neural layer as encoder and decoder

my_epochs = 100

# input = vecteur de taille 10
input_vect = Input(shape=(10,))

# "encoded" is the encoded representation of the inputs

encoded = Dense(6, activation='relu',)(input_vect)
encoded = Dense(3, activation='relu',)(encoded)

# "decoded" is the lossy reconstruction of the input

decoded = Dense(6, activation='relu')(encoded)
output = Dense(10, activation='linear')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_vect, output)

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1), loss='mse')

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


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


autoencoder.fit(x_train, x_train, epochs=my_epochs,batch_size=128, shuffle=False, validation_data=(x_test, x_test))