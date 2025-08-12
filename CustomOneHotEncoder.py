from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        """
        Initialise l'encodeur OneHot personnalisé pour une colonne donnée.
        
        Parameters:
        - column : str, nom de la colonne à encoder
        """
        self.column = column
        self.value_to_index = None

    def fit(self, X, y=None):
        """
        Apprend le dictionnaire de valeurs uniques à partir des données.
        
        Parameters:
        - X : DataFrame ou array-like, les données à utiliser pour apprendre l'encodeur.
        
        Retourne :
        - self : retourne l'instance de l'encodeur.
        """
        # Apprentissage des valeurs uniques dans la colonne
        self.value_to_index = {value: idx for idx, value in enumerate(X[self.column].unique())}
        return self

    def transform(self, X):
        """
        Transforme les données en vecteurs One-Hot.
        
        Parameters:
        - X : DataFrame ou array-like, les données à transformer.
        
        Retourne :
        - array : Un tableau NumPy avec les valeurs encodées en One-Hot.
        """
        # Création du tableau de One-Hot pour la colonne spécifiée
        one_hot_encoded = []
        for value in X[self.column]:
            one_hot_vector = [0] * len(self.value_to_index)
            if value in self.value_to_index:
                one_hot_vector[self.value_to_index[value]] = 1
            one_hot_encoded.append(one_hot_vector)
        
        # Convertir en DataFrame et concaténer avec les autres colonnes si besoin
        return pd.DataFrame(one_hot_encoded, columns=[f"{self.column}_{key}" for key in self.value_to_index])
