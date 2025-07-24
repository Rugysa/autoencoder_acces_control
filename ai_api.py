import keras
import sklearn
from keras.models import Model
from sklearn.preprocessing import StandardScaler
import numpy as np
from risk_utils import risk_utils



# Define a boolean as a tag : True is suspicious, False is normal
# Threshold is define in risk_utils
def label(score : float) :
  utils = risk_utils()
  return score > (utils.get_threshold())

# Receive data
# For now, we create data such as vectors with the same size that in train_autoencoder

# Base
x_model = np.array([0,1,2,3,4,5,6,7,8,9])
factor = 100 

# Here we creata data the same way that in train_autoencoder to see if the autoencoder works on positive data (affine relationship)
x_test = np.zeros((10,10000))
for i in range(10000) : 
  random_nb = np.random.random()
  bias = np.random.random()*factor
  x_test[:,i] = x_model*random_nb*factor + bias

# Here we creata negative data with an non-affine and decreasing relationship (the aim being to move away from the increasing affine learning relationship)
x_test_neg = np.zeros((10,10000))
for i in range(10000) : 
  random_nb = np.random.random()
  x_test_neg[:,i] = (1/(1+x_model)  )*random_nb*factor

# Preparation of data for the autoencoder
x_test = np.transpose(x_test)
x_test_neg = np.transpose(x_test_neg)
# Data centred and normalised
scaler = StandardScaler()
x_test = scaler.fit_transform(x_test)
x_test_neg = scaler.fit_transform(x_test_neg)

# Load the model from .keras
ai_engine = keras.models.load_model("model/autoencoder.keras")

# Prediction of the autoencoder
# output of positive data
x_predict = ai_engine.predict(x_test)
# output of negative data 
x_predict_neg = ai_engine.predict(x_test_neg)

# Console display of distances and tags
# Positive data
print('Dist positive data')
dist = sklearn.metrics.mean_squared_error(x_test, x_predict)
print(dist)
print(label(dist))

# Negative data
print('Dist negative data')
dist_neg = sklearn.metrics.mean_squared_error(x_test_neg, x_predict_neg)
print(dist_neg)
print(label(dist_neg))