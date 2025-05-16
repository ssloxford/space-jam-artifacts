import numpy as np
import matplotlib.pyplot as plt
import functools
import os

def normalise_power(signal, P=1):
    return signal * P/np.sqrt(measure_power(signal))

def to_dB(value):
    return 10*np.log10(value)

def from_dB(value):
    return 10**(value/10)

def measure_power(signal):
    return np.mean(np.abs(signal)**2)

plot_dir = "./plot/"
def plot(func):
    @functools.wraps(func)
    def wrapper(*args, output_pdf=None, dpi=None, show=True, **kwargs):
        res = func(*args, **kwargs)

        try:
            os.mkdir(plot_dir)
        except FileExistsError:
            pass

        if output_pdf is not None:
            print(f"saving to {output_pdf}")
            if dpi is None:
                plt.savefig(os.path.join(plot_dir, output_pdf), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(plot_dir, output_pdf), bbox_inches='tight', dpi=dpi)
        if show:
            plt.show()
        plt.close()

        return res

    return wrapper

def find_contiguous_true_blocks(bool_array, freq):
    # Ensure the input is a numpy array
    bool_array = np.asarray(bool_array)
    
    # Find the indices where the value changes
    change_indices = freq[np.where(np.diff(bool_array.astype(int)))[0]]
    return [(change_indices[i], change_indices[i + 1]) for i in range(0, len(change_indices) - 1, 2)]
