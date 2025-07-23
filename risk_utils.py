import numpy as np


class risk_utils :

    def __init__ (self) : 
        self.threshold = 10**(-3)

    def get_threshold(self) : 
        return self.threshold