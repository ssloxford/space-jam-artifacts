from scipy.signal import find_peaks, convolve, resample, butter, lfilter, welch
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from typing import Optional, Union
from sklearn.cluster import DBSCAN
from pathlib import Path

from lib.misc import from_dB, to_dB
from lib import Fourier, Signal
import lib.misc as misc

def gaussian_noise(n, N0, sample_rate):
    sigma = np.sqrt(N0/2)
    return Signal(
        iqs = np.random.normal(0,sigma,n) + 1j * np.random.normal(0,sigma,n),
        sample_rate=sample_rate
    )


def power_threshold(N, power_threshold):
    return N * power_threshold

def estimate_noise_floor(iqs, fft_magnitude):
    return np.mean(fft_magnitude[:iqs.size//10])

@dataclass
class NoiseDistribution:
    N0: float # TODO: refine this

    def cdf(self, p):
        raise NotImplementedError("subclass implements this")

    def plot_N0(self, ax, **kwargs):
        if ax is None:
            ax = plt.subplots()

        ax.axhline(self.N0, **kwargs)

@dataclass
class RayleighNoiseDistribution(NoiseDistribution):
    sigma: float
    variance: float

    def cdf(self, p):
        return sigma_rayleigh * np.sqrt(-2 * np.log(1-p))

@dataclass
class NoiseEstimator:
    def __call__(self, signal: Signal, fourier: Fourier) -> NoiseDistribution:
        raise NotImplementedError("subclass implements this")

class SliceNoiseEstimator(NoiseEstimator):
    # The section of the spectrum that we anticipate should be empty for noise estimation
    noise_estimation_proportion: float = 0.1

    def __call__(self, signal: Signal, fourier: Fourier):
        #print("DEBUG")
        #print(signal.iqs.size//(1//self.noise_estimation_proportion))

        noise_fft_magnitude = fourier.fft_magnitude[:signal.iqs.size//int(1//self.noise_estimation_proportion)] # This follows a Raleigh distribution
        sigma = np.sqrt(np.mean(noise_fft_magnitude**2)) # Scale parameter of Raleigh
        mean = sigma * np.sqrt(np.pi / 2)
        variance = (2 - np.pi / 2) * sigma**2

        return RayleighNoiseDistribution(
            N0 = mean,
            sigma = sigma,
            variance = variance
        )

@dataclass
class LabelledCarrier:
    frequency_low: float
    frequency_high: float
    frequency_centre: float
    #coarse_bandwidth_estimate: float

    def plot_frequency_low(self, ax, **kwargs):
        if ax is None:
            ax = plt.subplots()
        ax.axvline(self.frequency_low, **kwargs)

    def plot_frequency_high(self, ax, **kwargs):
        if ax is None:
            ax = plt.subplots()
        ax.axvline(self.frequency_high, **kwargs)

    def plot_frequency_centre(self, ax, **kwargs):
        if ax is None:
            ax = plt.subplots()
        ax.axvline(self.frequency_centre, **kwargs)

#    def plot_coarse_bandwidth_estimate(self, ax, **kwargs):
#        if ax is None:
#            ax = plt.subplots()
#        plt.fill_betweenx(800, low, high, where=(low <= freq) & (freq <= high), color='lightblue', alpha=0.5)

@dataclass
class CarrierDetector:
    pass

    def __call__(self, signal: Signal, fourier: Fourier, noise: NoiseDistribution) -> [LabelledCarrier]:
        raise NotImplementedError("subclass implements this")

@dataclass
class CarrierDetectorDebug:
    threshold: float
    peaks: np.ndarray
    conv: np.ndarray
    frequencies: np.ndarray

    def plot(self, ax, **kwargs):
        if ax is None:
            ax = plt.subplots()
        ax.semilogy(self.frequencies, self.conv)
        if self.peaks is not None:
            ax.plot(self.frequencies[self.peaks], self.conv[self.peaks], "x")
        ax.axhline(y=self.threshold, color="black", linestyle="dashed")

# TODO: make this its own class?
@dataclass
class WelchCarrierDetector:
    false_detection_rate: float
    nperseg: int = 256

    def __call__(self, signal: Signal, _: Fourier, __: NoiseDistribution) -> ([LabelledCarrier], CarrierDetectorDebug):
        frequencies, psd = welch(signal.iqs, fs=signal.sample_rate, nperseg=self.nperseg)

        psd = psd[np.argsort(frequencies)] # sort
        frequencies = frequencies[np.argsort(frequencies)] # sort

        data = np.column_stack((frequencies, np.abs(psd)))
        data[:, 1] = (data[:, 1] - np.min(data[:, 1])) / (np.max(data[:, 1]) - np.min(data[:, 1]))  # Normalize power

        frequency_resolution = signal.sample_rate / self.nperseg
        # Try using DBSCAN for clustering
        dbscan = DBSCAN(eps=800, min_samples=5)
        clusters = dbscan.fit_predict(data)
        print(clusters)

        # Plot the results
        plt.figure(figsize=(12, 6))
        #plt.semilogy(frequencies, psd, label='PSD')
        plt.plot(frequencies, psd, label='PSD')

#        # Plot clusters
#        unique_clusters = set(clusters)
#        for cluster in unique_clusters:
#            if cluster != -1:  # Ignore noise points
#                cluster_points = data[clusters == cluster]
#                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
#
#        plt.title('Power Spectral Density with DBSCAN Clustering')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.legend()
        plt.grid()
        plt.show()

        # Extract bandwidths for each cluster
        for cluster in unique_clusters:
            if cluster != -1:  # Ignore noise points
                cluster_points = data[clusters == cluster]
                frequency_range = (np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0]))
                print(f"Cluster {cluster}: Frequency range = {frequency_range[0]:.2f} Hz to {frequency_range[1]:.2f} Hz")

        import sys
        sys.exit()



        # Take a percentile based on the idea that we're oversampling
        #noise_power = np.mean(psd[psd < np.percentile(psd, 10)])  # Estimate noise power. TODO: use method from other block if it's Rayleigh
        #noise_std = np.std(psd[psd < np.percentile(psd, 10)])
        noise_power = np.mean(psd)
        noise_std = np.std(psd)
        threshold = noise_power + 0.5*noise_std
        #threshold = np.percentile(psd, (1-self.false_detection_rate)*100)
        #peaks, _ = find_peaks(psd, height=threshold)
        peak_frequencies = np.array(misc.find_contiguous_true_blocks(psd > threshold, frequencies))

        debug = CarrierDetectorDebug(
            threshold = threshold,
            peaks = None,
            conv = psd,
            frequencies = frequencies
        )

        results = []
        for (low, high) in peak_frequencies:
            results.append(
                LabelledCarrier(
                    frequency_low = low,
                    frequency_high = high,
                    frequency_centre = (high + low)/2
                    #coarse_bandwidth_estimate = high - low
                )
            )

        return results, debug

# TODO: pack this into the main dataclass
@dataclass
class ThresholdCarrierDetector(CarrierDetector):
    minimum_separation_bandwidth: float
    false_detection_rate: float = 0.001

    def __call__(self, signal: Signal, fourier: Fourier, noise: RayleighNoiseDistribution) -> ([LabelledCarrier], CarrierDetectorDebug):
        threshold = noise.sigma * np.sqrt(-2 * np.log(self.false_detection_rate))

        # Find the peaks
        peaks, _ = find_peaks(fourier.fft_magnitude, height=threshold)

        # Convolve the window to find contiguous peaks, discarding outliers
        window = np.ones(int(signal.sample_rate//self.minimum_separation_bandwidth))
        conv = (convolve(fourier.fft_magnitude, window, mode='same'))
        conv /= conv.max()
        peak_frequencies = np.array(misc.find_contiguous_true_blocks(conv > 0.3, fourier.freq))

        debug = CarrierDetectorDebug(
            threshold = threshold,
            peaks = peaks,
            conv = conv,
            frequencies = fourier.freq
        )

        results = []
        for (low, high) in peak_frequencies:
            results.append(
                LabelledCarrier(
                    frequency_low = low,
                    frequency_high = high,
                    frequency_centre = (high + low)/2
                    #coarse_bandwidth_estimate = high - low
                )
            )

        return results, debug

def frequency_shift(signal: Signal, frequency: float) -> Signal:
    t = np.arange(len(signal.iqs)) / signal.sample_rate
    carrier = np.exp(1j * 2 * np.pi * frequency * t)

    return Signal(
        iqs = signal.iqs * carrier,
        sample_rate = signal.sample_rate
    )

@dataclass
class BandwidthEstimate:
    bandwidth: float
    symbol_rate: float
    samples_per_symbol: float

@dataclass
class BandwidthEstimator:
    def __call__(self, signal: Signal, fourier: Fourier, labelled_carrier: LabelledCarrier) -> BandwidthEstimate:
        raise NotImplementedError("subclass implements this")

@dataclass
class CoarseBandwidthEstimator(BandwidthEstimator):
    def __call__(self, signal: Signal, _: Fourier, labelled_carrier: LabelledCarrier) -> BandwidthEstimate:
        bw = labelled_carrier.frequency_high - labelled_carrier.frequency_low
        return BandwidthEstimate(
            bandwidth = bw,
            symbol_rate = bw, # TODO: check we're not off by a factor of 2
            samples_per_symbol = signal.sample_rate/bw
        )

# This could be made generic
@dataclass
class FourierResampler:
    samples_per_symbol: int

    def __call__(self, signal: Signal, bandwidth_estimate: BandwidthEstimate) -> Signal:
        factor = self.samples_per_symbol / bandwidth_estimate.samples_per_symbol
        new_length = int(signal.iqs.size * factor)

        return Signal(
            iqs = resample(signal.iqs, new_length),
            sample_rate = signal.sample_rate * factor
        )

@dataclass
class Filter:
    order: int

    def __call__(self, signal: Signal, labelled_carrier: LabelledCarrier) -> Signal:
        # Shift the signal into positive frequencies, with a magic number margin
        margin = signal.sample_rate / 20
        shift_frequency = 0 if labelled_carrier.frequency_low > margin else margin - labelled_carrier.frequency_low
        shifted_signal = frequency_shift(signal, shift_frequency)

        # Create the filter
        nyq = 0.5 * shifted_signal.sample_rate
        low = (labelled_carrier.frequency_low + shift_frequency)/nyq
        high = (labelled_carrier.frequency_high + shift_frequency)/nyq
        # print("DEBUG")
        # print(f"low: {low}")
        # print(f"high: {high}")
        b, a = butter(self.order, [low, high], btype='band') #, fs=signal.sample_rate)

        # Apply Butterworth filter in time domain
        filtered_signal = lfilter(b, a, shifted_signal.iqs)

        # Shift into frequency domain
        freq_signal = np.fft.fft(filtered_signal)
        freq = np.fft.fftfreq(signal.iqs.size, d=1/signal.sample_rate)

        # Cancel negative frequencies
        filter_mask = np.zeros(signal.iqs.size, dtype=complex)
        filter_mask[freq >= 0] = 1
        filtered_freq_signal = freq_signal * filter_mask

        # Shift back to time domain
        filtered_signal = np.fft.ifft(filtered_freq_signal)

        # Shift back to original position
        deshifted_signal = frequency_shift(Signal(iqs=filtered_signal, sample_rate=signal.sample_rate), -shift_frequency)
        return deshifted_signal

@dataclass
class BasebandFilter:
    """
    NB: works assuming that the signal is at baseband
    This could be made generic later
    """
    order: int

    def __call__(self, signal: Signal, bandwidth_estimate: BandwidthEstimate):
        cutoff = bandwidth_estimate.bandwidth / 2
        # TODO: fix this
        #b, a = butter(self.order, [-cutoff / (0.5 * signal.sample_rate), cutoff / (0.5 * signal.sample_rate)], btype='band')
        #b, a = butter(self.order, [0, cutoff / (0.5 * signal.sample_rate)], btype='band')
        return Signal(
            #iqs = filtfilt(b, a, signal.iqs),
            iqs = signal.iqs,
            sample_rate = signal.sample_rate
        )

@dataclass
class ProcessedSignal:
    filtered_signal: Signal
    filtered_fourier: Fourier

    baseband_signal: Signal
    labelled_carrier: LabelledCarrier
    fourier: Fourier
    bandwidth_estimate: BandwidthEstimate

    resampled_signal: Signal
    resampled_fourier: Fourier

    normalised_signal: Signal
    normalised_fourier: Fourier

    # TODO: consider separating to make **kwargs apply separately to subplots?
    def plot_filtered_fourier(self, ax, **kwargs):
        if ax is None:
            ax = plt.subplots()

        self.filtered_fourier.plot(ax, **kwargs)

    def plot_baseband_fourier(self, ax, **kwargs):
        if ax is None:
            ax = plt.subplots()

        self.fourier.plot(ax, **kwargs)

        self.labelled_carrier.plot_frequency_low(ax, color="green", linestyle="dotted")
        self.labelled_carrier.plot_frequency_high(ax, color="red", linestyle="dotted")
        self.labelled_carrier.plot_frequency_centre(ax, color="black", linestyle="dotted")

    def plot_resampled_fourier(self, ax, **kwargs):
        if ax is None:
            ax = plt.subplots()

        self.resampled_fourier.plot(ax, **kwargs)

        self.labelled_carrier.plot_frequency_low(ax, color="green", linestyle="dotted")
        self.labelled_carrier.plot_frequency_high(ax, color="red", linestyle="dotted")
        self.labelled_carrier.plot_frequency_centre(ax, color="black", linestyle="dotted")

    def plot_normalised_fourier(self, ax, **kwargs):
        if ax is None:
            ax = plt.subplots()

        self.normalised_fourier.plot(ax, **kwargs)

        self.labelled_carrier.plot_frequency_low(ax, color="green", linestyle="dotted")
        self.labelled_carrier.plot_frequency_high(ax, color="red", linestyle="dotted")
        self.labelled_carrier.plot_frequency_centre(ax, color="black", linestyle="dotted")

@dataclass
class SignalProcessor:
    filter: Filter
    fourier_resampler: FourierResampler
    coarse_bandwidth_estimator: CoarseBandwidthEstimator = field(default_factory=CoarseBandwidthEstimator)

    def __call__(self, signal: Signal, labelled_carrier: LabelledCarrier) -> ProcessedSignal:
        # Filter the signal
        filtered_signal = self.filter(signal, labelled_carrier)
        # TODO: calculate fewer ffts, or on demand
        filtered_fourier = filtered_signal.fft()

        # Downconvert the signal
        baseband_signal = frequency_shift(filtered_signal, -labelled_carrier.frequency_centre)

        # Label the carrier
        labelled_carrier = LabelledCarrier(
            frequency_low = labelled_carrier.frequency_low - labelled_carrier.frequency_centre,
            frequency_high = labelled_carrier.frequency_high - labelled_carrier.frequency_centre,
            frequency_centre = 0,
        )

        # Take the fourier transform
        fourier = baseband_signal.fft()

        # Perform coarse bandwidth estimation
        coarse_bandwidth_estimate = self.coarse_bandwidth_estimator(signal, fourier, labelled_carrier)

        # TODO: perform filter matching

        # Resample to the required symbol rate
        resampled_signal = self.fourier_resampler(baseband_signal, coarse_bandwidth_estimate)
        resampled_fourier = resampled_signal.fft()

        normalised_signal = resampled_signal.scale_power()
        normalised_fourier = normalised_signal.fft()

        return ProcessedSignal(
            baseband_signal = baseband_signal,
            labelled_carrier = labelled_carrier,
            fourier = fourier,
            bandwidth_estimate = coarse_bandwidth_estimate,
            resampled_signal = resampled_signal,
            resampled_fourier = resampled_fourier,
            filtered_signal = filtered_signal,
            filtered_fourier = filtered_fourier,
            normalised_signal = normalised_signal,
            normalised_fourier = normalised_fourier
        )

@dataclass
class PreprocessorOutput:
    fourier: Fourier
    noise: NoiseDistribution
    labelled_carriers: [LabelledCarrier]
    processed_signals: [ProcessedSignal]
    carrier_detector_debug: CarrierDetectorDebug

    # TODO: consider separating to make **kwargs apply separately to subplots?
    def plot_fourier(self, ax, **kwargs):
        if ax is None:
            ax = plt.subplots()

        self.fourier.plot(ax)

        for carrier in self.labelled_carriers:
            carrier.plot_frequency_low(ax, color="green", linestyle="dotted")
            carrier.plot_frequency_high(ax, color="red", linestyle="dotted")
            carrier.plot_frequency_centre(ax, color="black", linestyle="dotted")

@dataclass
class Preprocessor:
    noise_estimator: NoiseEstimator
    carrier_detector: CarrierDetector
    signal_processor: SignalProcessor

    def __call__(self, signal: Union[[Signal], Signal]) -> Union[[PreprocessorOutput], PreprocessorOutput]:
        if isinstance(signal, list):
            outputs = []
            for chunk in signal:
                outputs.append(self(chunk))
            return outputs

        fourier = signal.fft()
        noise = self.noise_estimator(signal, fourier)
        labelled_carriers, carrier_detector_debug = self.carrier_detector(signal, fourier, noise)

        processed_signals = [self.signal_processor(signal, labelled_carrier) for labelled_carrier in labelled_carriers]

        return PreprocessorOutput(
            fourier = fourier,
            noise = noise,
            labelled_carriers = labelled_carriers,
            processed_signals = processed_signals,
            carrier_detector_debug = carrier_detector_debug
        )
