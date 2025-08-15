from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column
        self.value_to_index = None

    # Learns the dictionary of unique values from the data and return the encoder
    def fit(self, df, y=None):
        # Learning unique values in the column
        self.value_to_index = {value: idx for idx, value in enumerate(df[self.column].unique())}
        return self

    # Converts data into One-Hot vectors
    def transform(self, df):
        # Create a One-Hot table for the specified column
        one_hot_encoded = []
        for value in df[self.column]:
            one_hot_vector = [0] * len(self.value_to_index)
            if value in self.value_to_index:
                one_hot_vector[self.value_to_index[value]] = 1
            one_hot_encoded.append(one_hot_vector)

        df_copy = df.copy()
        df_copy[self.column]= one_hot_encoded
        return df_copy

    # Return the dictionary
    def get_dico(self) :
        return self.value_to_index
