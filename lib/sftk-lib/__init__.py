import numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Fourier:
    fft: np.ndarray # [np.csingle]
    fft_magnitude: np.ndarray # [np.csingle]
    freq: np.ndarray # [np.csingle]

    def plot(self, ax, **kwargs):
        if ax is None:
            ax = plt.subplots()

        ax.plot(self.freq, self.fft_magnitude)

# TODO: consider making Signal a generator of chunks
#   Or a subclass StreamingSignal
@dataclass
class Signal:
    """Represents a raw I/Q signal, ready for processing"""
    iqs: np.ndarray
    sample_rate: int

    def plot(self, ax, n=200, **kwargs):
        if ax is None:
            ax = plt.subplots()
        ax.plot(self.iqs[:200], **kwargs)

    def power(self):
        return np.mean(np.abs(self.iqs)**2)

    def from_file(self, path: Path):
        pass

    def __str__(self):
        return f"Signal(sample_rate={sample_rate}, size={self.iqs.size})"

    def __len__(self):
        return self.iqs.size

    def __add__(self, signal): # TODO: add a type
        if type(signal) is not self.__class__:
            raise RuntimeError("signals con only be added to other signals")

        if self.sample_rate != signal.sample_rate:
            raise RuntimeError("mismatched sample rate")

        return Signal(
            iqs = self.iqs + signal.iqs,
            sample_rate = self.sample_rate
        )

    def __sub__(self, signal): # TODO: add a type
        if type(signal) is not self.__class__:
            raise RuntimeError("signals con only be added to other signals")

        if self.sample_rate != signal.sample_rate:
            raise RuntimeError("mismatched sample rate")

        return Signal(
            iqs = self.iqs - signal.iqs,
            sample_rate = self.sample_rate
        )

    def __getitem__(self, slice):
        return Signal(
            iqs = self.iqs[slice],
            sample_rate = self.sample_rate
        )

    def chunk(self, n: int): # -> [self.__class__]:
        max_elements = (len(self)//n) * n
        truncated_signal = self[:max_elements]
        chunks = truncated_signal.iqs.reshape(-1, n)
        return [Signal(iqs=chunk, sample_rate=self.sample_rate) for chunk in chunks]

    def fft(self) -> Fourier:
        _fft = np.fft.fft(self.iqs)
        freq = np.fft.fftfreq(self.iqs.size, d=1/self.sample_rate)
        _fft = _fft[np.argsort(freq)] # sort
        freq = freq[np.argsort(freq)] # sort
        fft_magnitude = np.abs(_fft)

        return Fourier(
            fft = _fft,
            freq = freq,
            fft_magnitude = fft_magnitude
        )

    def scale_power(self, factor: float = 1):
        return Signal(
            iqs = (self.iqs / np.sqrt(np.mean(np.abs(self.iqs)**2))) * factor,
            sample_rate = self.sample_rate
        )
