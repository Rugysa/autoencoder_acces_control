import keras
import sklearn
from keras.models import Model
from sklearn.preprocessing import StandardScaler
import numpy as np
from risk_utils import risk_utils
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder



# Define a boolean as a tag : True is suspicious, False is normal
# Threshold is define in risk_utils
def label(score : float) :
  utils = risk_utils()
  return score > (utils.get_threshold())

# Receive data

# Charger le CSV
df = pd.read_csv("full_dataset.csv")

# Nettoyer les noms de colonnes
df.columns = [col.strip() for col in df.columns]

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


# Colonnes numériques et catégorielles
num_cols = ['session_duration', 'power_usage', 'lat', 'lon',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
cat_cols = ['role', 'location', ]


# Enleve les données normales
df_neg = df[df['access_granted'] == False].copy()
# Enleve les données anormales
df_pos = df[df['access_granted'] == True].copy()


# Supprimer access_granted (utilisé plus tard pour l'évaluation)
df_pos.drop(columns=['access_granted'], inplace=True) # suppression car non supervisé (entriané que sur données normales, mais du coup ici on a des donnes anormales)
# Supprimer access_granted (utilisé plus tard pour l'évaluation)
df_neg.drop(columns=['access_granted'], inplace=True) # suppression car non supervisé (entriané que sur données normales, mais du coup ici on a des donnes anormales)


# Pipeline de prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ]
)

# Transformation
X = preprocessor.fit_transform(df_pos)
X_neg = preprocessor.fit_transform(df_neg)


"""
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
"""

# Load the model from .keras
ai_engine = keras.models.load_model("model/autoencoder.keras")

# Prediction of the autoencoder
# output of positive data
x_predict = ai_engine.predict(X)
# output of negative data 
x_predict_neg = ai_engine.predict(X_neg)

# Console display of distances and tags
# Positive data
print('Dist positive data')
dist = sklearn.metrics.mean_squared_error(X, x_predict)
print(dist)
print(label(dist))

# Negative data
print('Dist negative data')
dist_neg = sklearn.metrics.mean_squared_error(X_neg, x_predict_neg)
print(dist_neg)
print(label(dist_neg))