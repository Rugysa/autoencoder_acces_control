import pandas as pd
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from CustomOneHotEncoder import CustomOneHotEncoder

class data_treatment : 

    def __init__ (self,path_input) :
        self.path = path_input
        self.df = None

    def load_csv (self) : 
        self.df = pd.read_csv(self.path)
        return self.df

    def clean_dataFrame (self) :
        # Clean column names
        self.df.columns = [col.strip() for col in self.df.columns]
        # Remove columns that are not necessary for learning
        self.df.drop(columns=['ev_user_id', 'charger_id', 'behavior_context','risk_score'], inplace=True)

    # Modify the timestamp into usable data
    def timestamp_treatment (self) : 
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['dow_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['dow_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        self.df.drop(columns=['timestamp', 'hour', 'day_of_week'], inplace=True)
    
    # Split geographical coordinates into latitudes and longitudes
    def coordonates_treatment (self) :
        self.df[['lat', 'lon']] = self.df['geo_coordinates'].str.split(',', expand=True).astype(float)
        self.df.drop(columns=['geo_coordinates'], inplace=True)

    # Initialises the dictionaries of the custom OneHotEncoders
    def init_customOneHotEncoder (self) :
        # Initisalisation of the location encoder
        self.encoder_location = CustomOneHotEncoder(column='location')
        self.encoder_location.fit(self.df)
        # Initisalisation of the role encoder
        self.encoder_role = CustomOneHotEncoder(column='role')
        self.encoder_role.fit(self.df)

    # Keep the normal or anoumalous data according to keep_true
    def filer_data(self, keep_true) :
        # Keep the data
        new_df = self.df[self.df['access_granted'] == keep_true].copy()
        # Remove data with access_granted (unsupervised learning)
        new_df.drop(columns=['access_granted'], inplace=True)
        return new_df

    # Processing text columns with the custom OneHotEncoder
    # It is necessary to have a dataframe as input because we no longer work with self.data 
    # (the previous function returned a modified copy, which is what we use).
    def textual_data_treatment (self,input_df) :
        input_df = self.encoder_location.transform(input_df)
        input_df = self.encoder_role.transform(input_df)
        return input_df

    # Processing numerical columns and transform the dataframe into a vector for the autoencoder
    def numerical_data_treatment (self,input_df) :
        num_cols = ['session_duration', 'power_usage', 'lat', 'lon',
                    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), num_cols)
            ]
        )
        X = self.preprocessor.fit_transform(input_df)
        return X
    