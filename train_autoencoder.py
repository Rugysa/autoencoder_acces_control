from autoencoder import autoencoder
from data_treatment import data_treatment
import sklearn
from sklearn.model_selection import train_test_split


# Parameters definition
my_epochs = 60
input_length = 8


# Import data
data_treatment = data_treatment(path_input="full_dataset.csv")
data_treatment.load_csv()

# Data treatment
data_treatment.clean_dataFrame()
data_treatment.timestamp_treatment()
data_treatment.coordonates_treatment()
data_treatment.init_customOneHotEncoder()
df_pos = data_treatment.filer_data(True)
df_pos = data_treatment.textual_data_treatment(df_pos)
X = data_treatment.numerical_data_treatment(df_pos)


# Split train/val
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# Train of the autoencoder from autoencoder.py
autoencoder = autoencoder(my_epochs,input_length)
autoencoder.compile()
autoencoder.model.fit(X_train, X_train, epochs=my_epochs,batch_size=128, shuffle=False, validation_data=(X_val, X_val))


# Save parameters of the autoencoder in autoencoder.keras
autoencoder.save()