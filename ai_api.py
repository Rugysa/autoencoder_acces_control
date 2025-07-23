import keras
import sklearn
from keras.models import Model
from sklearn.preprocessing import StandardScaler
import numpy as np
from risk_utils import risk_utils



# Define a boolean as a tag : True is suspicious, False is normal
def label(score : float) :
  utils = risk_utils()
  return score > (utils.get_threshold())

# receive data
# for now, we create data such as vectors
x_test = np.zeros((10,10000)) #ça me parait plus propre
x_model = np.array([0,1,2,3,4,5,6,7,8,9])
factor = 100  
for i in range(10000) : 
  random_nb = np.random.random() # génère un flotant aléatoire entre 0 et 1
  bias = np.random.random()*factor #je pense que faire un truc qu'affine plutôt que linéaire peut être interessant
  x_test[:,i] = x_model*random_nb*factor + bias

x_test_neg = np.zeros((10,10000)) #ça me parait plus propre
for i in range(10000) : 
  random_nb = np.random.random() # génère un flotant aléatoire entre 0 et 1
  x_test_neg[:,i] = (1/(1+x_model)  )*random_nb*factor

x_test = np.transpose(x_test)
x_test_neg = np.transpose(x_test_neg)

scaler = StandardScaler()

x_test = scaler.fit_transform(x_test)
x_test_neg = scaler.fit_transform(x_test_neg)


ai_engine = keras.models.load_model("model/autoencoder.keras")
x_predict = ai_engine.predict(x_test)
x_predict_neg = ai_engine.predict(x_test_neg)

print('dist positive data')
dist = sklearn.metrics.mean_squared_error(x_test, x_predict)
print(dist)
print(label(dist))

print('dist negative data')
dist_neg = sklearn.metrics.mean_squared_error(x_test_neg, x_predict_neg)
print(dist_neg)
print(label(dist_neg))
