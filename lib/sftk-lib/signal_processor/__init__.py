import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import lib.jamming_ber as jamming_ber
from lib import Signal

class SignalProcessor():
    def __call__(self, signal: Signal):
        raise NotImplementedError("subclass implements this")

    def power(self):
        """
        Returns the 
        """
        raise NotImplementedError("subclass implements this")

    def name(self):
        raise NotImplementedError("subclass implements this")

class AWGN(SignalProcessor):
    def __init__(self, N0):
        self.N0 = N0
        
    def __call__(self, iqs):
        noise_iq = jamming_ber.gaussian_noise(iqs.shape, self.N0)
        iqs += noise_iq
        return iqs

    def name(self):
        return f"{self.N0}-AWGN"
