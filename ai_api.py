import keras
import sklearn
from data_treatment import data_treatment

# Define a boolean as a tag : True is suspicious, False is normal
def label(score : float) :
  threshold = 10**(-2) # Arbitrary choice based on several uses of the autoencoder
  return score > threshold

# Receive data
data_treatment = data_treatment(path_input="full_dataset.csv")
data_treatment.load_csv()

# Data treatment
data_treatment.clean_dataFrame()
data_treatment.timestamp_treatment()
data_treatment.coordonates_treatment()
data_treatment.init_customOneHotEncoder()

# Recovery and specific treatment of positive data
df_pos = data_treatment.filer_data(True)
df_pos = data_treatment.textual_data_treatment(df_pos)

# Recovery and specific treatment of negative data
df_neg = data_treatment.filer_data(False)
df_neg = data_treatment.textual_data_treatment(df_neg)

# numerical treatment is the same for every data type
data_treatment.numerical_data_treatment()

X = data_treatment.final_transformation(df_pos)
X_neg = data_treatment.final_transformation(df_neg)

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