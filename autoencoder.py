import keras
from keras.layers import Input, Dense
from keras.models import Model


# input dimension -> type of layer -> output dimension
# here :  
# Autoencoder consistinf of : 8 -> Dense -> 8 -> Dense -> 4 -> Dense -> 8 -> Dense -> 8
# Autoencoer define by number of epoch (nb_epochs) and the dimension of the input (input_length)
class autoencoder :

  def __init__(self,nb_epochs : int, input_length : int) :
    self.epochs = nb_epochs
    self.input = Input(shape=(input_length,))
    self.input_size = input_length
    self.latent_space_size = 4
    self.inter_layer_size = 8

  # Definition of encoder
  # 8 -> Dense -> 8 -> Dense -> 4
  # Return the encoder model's if necessary
  def encoder(self):
    encoded = Dense(self.inter_layer_size , activation='relu',)(self.input)
    self.output_encoder = Dense(self.latent_space_size, activation='relu',)(encoded)
    self.encoder = Model(self.input, self.output_encoder)
    return self.encoder

  # Definition of decoder
  # 4 -> Dense -> 8 -> Dense -> 8
  # Return the decoder model's if necessary
  def decoder(self): 
    decoded = Dense(self.inter_layer_size , activation='relu')(self.output_encoder)
    self.output = Dense(self.input_size, activation='linear')(decoded)
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
    self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

  def save(self) :
    # Using Keras
    self.model.save("model/autoencoder.keras")