import keras
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from sklearn.preprocessing import StandardScaler


# input dimension -> type of layer -> output dimension
# here :  
# Autoencoder consistinf of : 10 -> Dense -> 6 -> Dense -> 2 -> Dense -> 6 -> Dense -> 10
# Autoencoer define by number of epoch (nb_epochs) and the dimension of the input (input_length)
class autoencoder :

  def __init__(self,nb_epochs : int, input_length : int) :
    self.epochs = nb_epochs
    self.input = Input(shape=(input_length,))

  # Definition of encoder
  # 10 -> Dense -> 6 -> Dense -> 2
  # Return the encoder model's if necessary
  def encoder(self):
    encoded = Dense(10, activation='relu',)(self.input)
    self.output_encoder = Dense(3, activation='relu',)(encoded)
    self.encoder = Model(self.input, self.output_encoder)
    return self.encoder

  # Definition of decoder
  # 2 -> Dense -> 6 -> Dense -> 10
  # Return the decoder model's if necessary
  def decoder(self): 
    decoded = Dense(10, activation='relu')(self.output_encoder) 
    self.output = Dense(16, activation='linear')(decoded)
    self.decoder = Model(self.output_encoder, self.output)
    return self.decoder

  # Record and return the autoencoder model's
  def model(self) : 
    self.model = Model(self.input, self.output)
    return self.model

  # Function call by train_autoencoder : build and compile the autoencoder
  def compile(self) :
    self.encoder()
    self.decoder()
    self.model()
    self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')

  def save(self) :
    # Using Keras
    self.model.save("model/autoencoder.keras")
    # Using PyTorch
    #torch.save(self.model.state_dict(), 'autoencoder.pth')