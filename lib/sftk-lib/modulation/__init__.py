import numpy as np
import numpy.ma as ma
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import lib.jamming_ber as jamming_ber

# Parent class of modulation schemes
# All modulations must have average symbol energy normalized to 1

# TODO: model the probability distribution of this directly, as well as the mean and variance
class Modulation():
    symbols: np.array = None
    bits: np.array = None # TODO: consider getting rid of this, and ordering symbols in bit-order

    def name(self):
        raise NotImplementedError("subclass implements this")
        
    def symbols(self):
        assert(self.symbols is not None)
        return symbols
        
    def bits(self):
        assert(self.bits is not None)
        return bits
    
    def n_symbols(self):
        return len(self.symbols)
    
    def bits_per_symbol(self) -> int:
        return int(np.log2(len(self.symbols)))

    def distances(self, n_differing_bits=None) -> np.array:
        """
        Returns the distances between symbols in the constellation scheme.
        Filters the resulting distances to include only symbols which differ in n_differing_bits
        """
        symbs = self.symbols
        distances = np.abs(symbs[:, np.newaxis] - symbs[np.newaxis, :]) # Calculate pairwise distances
        np.fill_diagonal(distances, None) # Ignore zero distances

        if n_differing_bits is None:
            return distances

        mask = np.zeros(distances.shape, dtype=distances.dtype)

        for i, vi in enumerate(self.bits):
            for j, vj in enumerate(self.bits):
                if bin(vi ^ vj).count('1') != n_differing_bits:
                    mask[i][j] = np.inf

        return distances + mask

    def min_distance(self, n_differing_bits=None) -> float:
        """
        Returns the minimum distance of this modulation scheme
        """
        return np.min(ma.masked_invalid(self.distances(n_differing_bits=n_differing_bits)))

    def max_distance(self, n_differing_bits=None) -> float:
        """
        Returns the maximum distance of this modulation scheme
        """
        return np.max(ma.masked_invalid(self.distances(n_differing_bits=n_differing_bits)))
    
    # TODO: consider replacing with modulate, demodulate functions
    # Returns a generator function of modulated (symbols, bits), according to weights
    def generate_symbols(self, symbol_weights, size):
        choices = np.random.choice(np.arange(0, self.n_symbols()), size=size, p=symbol_weights)
        return (self.symbols[choices], self.bits[choices])

    def modulate(self, data):
        if type(data) != int:
            raise NotImplementedError("Currently can only modulate integer types")
        # Slice data into an array where each element has self.bits_per_symbol() bits
        binary_data = bin(data)[2:]

        bits_per_symbol = self.bits_per_symbol()
        padded_length = len(binary_data) + (bits_per_symbol - len(binary_data) % bits_per_symbol) % bits_per_symbol
        binary_data = binary_data.zfill(padded_length)

        # Create a NumPy array to hold the symbols
        num_symbols = len(binary_data) // bits_per_symbol
        symbols = np.zeros(num_symbols, dtype=np.csingle)
        
        # Slice the binary data into chunks of bits_per_symbol and convert to integers
        for i in range(num_symbols):
            symbol = binary_data[i * bits_per_symbol:(i + 1) * bits_per_symbol]
            symbols[i] = self.symbols[self.bits[int(symbol, 2)]]
        return symbols
    
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(np.real(self.symbols), np.imag(self.symbols), c=range(len(self.symbols)))
        n = max(map(lambda x: len(bin(x)[2:]), self.bits))
        for c, b in zip(self.symbols, self.bits):
            ax.text(c.real, c.imag, format(b, f"0{n}b"))
        #ax = plt.gca()
        ax.axis('equal')

# Generates a Gray-coded PSK constellation
class PSK(Modulation):
    def __init__(self, M: int):
        # TODO: rename these to something nice
        constellation, _ = jamming_ber.norm_constellation(np.exp(1j * np.linspace(0, 2*np.pi, M, endpoint=False)))
        symbol_ids = np.array(range(M))
        symbol_bits = symbol_ids ^ (symbol_ids >> 1) # gray coding
        
        self.symbols = constellation
        self.bits = symbol_bits
        self.M = M
        
    def name(self):
        if self.M == 2:
            return "BPSK"
        elif self.M == 4:
            return "QPSK"
        else:
            return f"{self.M}-PSK"

# Generates a custom-labelled APSK constellation
# TODO: consider simplifying the arguments that are passed in
class APSK(Modulation):
    def __init__(self, bit_labels, radii, angular_offsets):
        constellation = []
        labels = []

        for L, R, A in zip(bit_labels, radii, angular_offsets):
            M = len(L)
            cons = jamming_ber.generate_psk_constellation(M)[0]

            # Scale and rotate the constellation
            cons = map(lambda x: x * (np.cos(A) + np.sin(A)*1j) * R, cons)

            # Apply the labels
            constellation += cons
            labels += L

        (cons, factor) = jamming_ber.norm_constellation(np.array(constellation))
        self.symbols = cons
        self.bits = np.array(labels)
        self.bit_labels = bit_labels
        
    # TODO: implement this unambiguously for each of the constellation variants e.g. 8-APSK but with different radii
    def name(self):
        n_bit_labels = np.sum(list(map(len, self.bit_labels)))
        M = int(np.log2(n_bit_labels))
        return f"{M}-APSK"

class QAM(Modulation):
    def __init__(self, M: int):
        sqrtM = int(np.sqrt(M))
        bits_per_side = int(np.log2(sqrtM))
        sides = np.linspace(-1, 1, sqrtM, endpoint=True)
        constellation_i, constellation_q = np.meshgrid(sides, sides)
        constellation, factor = jamming_ber.norm_constellation((constellation_i + 1j*constellation_q).flatten())

        symbol_ids = np.array(range(sqrtM))
        symbol_bits = symbol_ids ^ (symbol_ids >> 1)
        symbol_bits_r, symbol_bits_c = np.meshgrid(symbol_bits, symbol_bits)
        symbol_bits_full = (symbol_bits_r.flatten() << bits_per_side) + symbol_bits_c.flatten()

        self.symbols = constellation
        self.bits = symbol_bits_full
