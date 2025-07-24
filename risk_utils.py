import numpy as np


class risk_utils :

    def __init__ (self) : 
        self.threshold = 10**(-3) # Arbitrary choice based on several uses of the autoencoder

    # Return the threshold
    def get_threshold(self) : 
        return self.threshold