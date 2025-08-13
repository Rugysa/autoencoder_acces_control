from autoencoder import autoencoder
from CustomOneHotEncoder import CustomOneHotEncoder
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model

# Parameters definition
my_epochs = 30
input_length = 8


# Import data


# Charger le CSV
df = pd.read_csv("full_dataset.csv")

# Nettoyer les noms de colonnes
df.columns = [col.strip() for col in df.columns]

# Enleve les données anormales
df = df[df['access_granted'] == True].copy()

# Enelve les colonnes inutiles à l'apprentissage
df.drop(columns=['ev_user_id', 'charger_id', 'behavior_context','risk_score'], inplace=True) 


# Transformer le timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df.drop(columns=['timestamp', 'hour', 'day_of_week'], inplace=True)

# Séparer geo_coordinates en lat/lon
df[['lat', 'lon']] = df['geo_coordinates'].str.split(',', expand=True).astype(float)
df.drop(columns=['geo_coordinates'], inplace=True)

encoder = CustomOneHotEncoder(column='location')
encoder.fit(df)


encoder_role = CustomOneHotEncoder(column='role')
encoder_role.fit(df)

# Supprimer access_granted (utilisé plus tard pour l'évaluation)
df.drop(columns=['access_granted'], inplace=True) # suppression car non supervisé (entriané que sur données normales, mais du coup ici on a des donnes anormales)

# Transforme la colonne catégorielle avant le pipeline
one_hot_transformed = encoder.transform(df)
one_hot_transformed_role = encoder_role.transform(df)

# Ajouter les colonnes One-Hot transformées au DataFrame original (si nécessaire)
df = pd.concat([df, one_hot_transformed], axis=1)
df = pd.concat([df, one_hot_transformed_role], axis=1)



# Colonnes numériques et catégorielles
num_cols = ['session_duration', 'power_usage', 'lat', 'lon',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
cat_cols = ['role']
loc_col = ['location']

# Vérification rapide
assert all(col in df.columns for col in num_cols + cat_cols), "Colonne manquante !"





# Pipeline de prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), num_cols)
    ]
)










# Transformation
X = preprocessor.fit_transform(df)

# Split train/val
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
input_dim = X.shape[1]
print(input_dim)
"""
input_layer = Input(shape=(input_length,))

# Encodeur (réduction de la dimension)
encoded = Dense(64, activation='relu')(input_layer)  # Augmenter la largeur de la couche cachée
encoded = Dropout(0.2)(encoded)  # Dropout pour éviter le surapprentissage
encoded = Dense(32, activation='relu')(encoded)  # Réduction de la dimension encore plus faible

# Décodeur (reconstruction de la donnée originale)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dropout(0.2)(decoded)  # Dropout pour éviter le surapprentissage
decoded = Dense(input_dim, activation='linear')(decoded)  # Activation linéaire pour reconstruire l'entrée


model = Model(input_layer, decoded)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entraînement du modèle
history = autoencoder.fit(X_train, X_train,
                          epochs=my_epochs,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(X_val, X_val),
                          callbacks=[early_stopping],  # Ajout du callback EarlyStopping
                          verbose=1)

model.fit(X_train, X_train, epochs=my_epochs,batch_size=128, shuffle=False, validation_data=(X_val, X_val))
"""

'''
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
'''


# Train of the autoencoder from autoencoder.py
autoencoder = autoencoder(my_epochs,input_length)
autoencoder.compile()
autoencoder.model.fit(X_train, X_train, epochs=my_epochs,batch_size=128, shuffle=False, validation_data=(X_val, X_val))


# Save parameters of the autoencoder in autoencoder.pth (PyTorch)
# Here : use of .keras because we use Keras and not PyTorch
autoencoder.save()