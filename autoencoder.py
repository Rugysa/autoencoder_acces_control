import keras
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from sklearn.preprocessing import StandardScaler


class autoencoder :

  def __init__(self,nb_epochs : int, input_length : int) :
    self.epochs = nb_epochs
    self.input = Input(shape=(input_length,))


  def encoder(self):
    encoded = Dense(6, activation='relu',)(self.input)
    self.output_encoder = Dense(2, activation='relu',)(encoded)
    self.encoder = Model(self.input, self.output_encoder)
    return self.encoder


  def decoder(self): 
    decoded = Dense(6, activation='relu')(self.output_encoder) 
    self.output = Dense(10, activation='linear')(decoded)
    self.decoder = Model(self.output_encoder, self.output)
    return self.decoder


  def model(self) : 
    self.model = Model(self.input, self.output)
    return self.model

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