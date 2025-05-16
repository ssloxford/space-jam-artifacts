import os
# Eventually when these deps are full packages, we won't need to update the env like this
# Set the PYTHONPATH
os.environ['PYTHONPATH'] = '/usr/local/lib/python3.12/site-packages/' + os.environ.get('PYTHONPATH', '')

# Set the LD_LIBRARY_PATH
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib' + os.environ.get('LD_LIBRARY_PATH', '')

# TODO: clean up unused imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tempfile import NamedTemporaryFile
import itertools
from tqdm import tqdm
import random
from functools import partial
from random import randint
import subprocess
from dataclasses import dataclass, asdict, field

from lib.modulation.dvb_s2 import BPSK, QPSK, PSK_8, APSK_8_rate_100_180, APSK_16_rate_5_6, APSK_32_rate_8_9

## Allow these to be used in other projects
try:
    from ldpc_encoder import ldpc_encoder
except ImportError as e:
    print(e)
try:
    from ldpc_ng_2 import ldpc_ng_2 as ldpc_ng
except ImportError as e:
    print(e)
from lib.jamming_ber import gaussian_noise
from lib.misc import from_dB, to_dB

# Set up the error correcting codes
codes = [
    {
        "alist": "alists/ccsds_tc_64.alist",
        "n": 128,
        "k": 64,
        "name": "CCSDS TC",
        "limits": {
            ("desync", "gaussian"): (-2.0, 4.5),
            ("desync", "symbol"): (-1.0, 4.5),
            ("sync", "symbol"): (-3.0, 3.0)
        },
        "max_iterations": 100
    },
    {
        "alist": "alists/ccsds_tc_256.alist",
        "n": 512,
        "k": 256,
        "name": "CCSDS TC",
        "limits": {
            ("desync", "gaussian"): (-0.5, 4.0),
            ("desync", "symbol"): (-0.5, 4.0),
            ("sync", "symbol"): (-3.0, 3.0)
        },
        "max_iterations": 100
    },
    {
        "alist": "alists/ar4ja_4_5_1024.alist",
        "n": 1408,
        "k": 1024,
        "name": "CCSDS TM AR4JA",
        "limits": {
            ("desync", "gaussian"): (-7.0, -3.0),
            ("desync", "symbol"): (-6.0, -1.0),
            ("sync", "symbol"): (-6.0, -2.0)
        },
        "max_iterations": 100
    },
    {
        "alist": "alists/dvbs2_9_10.alist",
        "n": 64800,
        "k": 58320,
        "name": "DVB-S2",
        "limits": {
            ("desync", "gaussian"): (-8.0, -5.0),
            ("desync", "symbol"): (-6.0, -3.0),
            ("sync", "symbol"): (-7.5, -4.0)
        },
        "max_iterations": 25
    },
    {
        "alist": "alists/dvbs2_1_4.alist",
        "n": 64800,
        "k": 16200,
        "name": "DVB-S2",
        "limits": {
            ("desync", "gaussian"): (1.0, 4.0), # Will be interesting to see if these turn out the same
            ("desync", "symbol"): (1.0, 4.0), # TODO: check this
            ("sync", "symbol"): (1.0, 4.0)
        },
        "max_iterations": 25
    },

    { # 5
        "alist": "alists/dvbs2_4_5.alist",
        "n": 64800,
        "k": 51840,
        "name": "DVB-S2",
        "limits": {
            ("desync", "gaussian"): (1.0, 4.0), # Will be interesting to see if these turn out the same
            ("desync", "symbol"): (1.0, 4.0), # TODO: check this
            ("sync", "symbol"): (1.0, 4.0)
        },
        "max_iterations": 25
    },

    { # 6
        "alist": "alists/dvbs2_3_4.alist",
        "n": 64800,
        "k": 48600,
        "name": "DVB-S2",
        "limits": {
            ("desync", "gaussian"): (1.0, 4.0), # Will be interesting to see if these turn out the same
            ("desync", "symbol"): (1.0, 4.0), # TODO: check this
            ("sync", "symbol"): (1.0, 4.0)
        },
        "max_iterations": 25
    },
]


# Wrapper function to encode LDPC frames
def encode(code, n_frames, random_frames=True):
    with NamedTemporaryFile() as out_file:
        tb = ldpc_encoder(
            alist = code["alist"],
            k = code["k"],
            n = code["n"],
            n_frames = n_frames,
            out_file = out_file.name,
            random_frames = 1 if random_frames else 0,
            seed=randint(0, 65535)
        )

        tb.start()
        tb.wait()

        encoded_frames = np.fromfile(out_file.name, dtype=np.uint8)
        return encoded_frames

# Modulation
def mod_bpsk(data):
    return BPSK.symbols[data]

QPSK_symbols = QPSK.symbols * np.exp(1j * (np.pi/4))
def mod_qpsk(data):
    # Batch data into two bits
    x = data.reshape(-1, 2)
    x[:, 0] *= 2
    x=x.sum(axis=1)
    idx = QPSK.bits[x]

    return QPSK_symbols[idx]

# Demodulation
def demod_bpsk(samples: np.ndarray, noise_sigma: float):
    # Negative scale because we use the convention that +1 means a 1
    # bit.
    scale = -2.0 / (noise_sigma**2)
    return (samples.real * scale)

def demod_qpsk(samples: np.ndarray, noise_sigma: float):
    scale = (-2.0 * (1/np.sqrt(2))) / (noise_sigma**2)
    return np.stack([np.imag(samples), np.real(samples)], axis=-1) * scale    

# TODO: finish this
def decode_new(data, code, max_iterations=100, decoder="Phif64"):
    # Convert data to float32
    data_f32 = data.astype(np.float32) * -1 # Hack: somehow the decoder we're using requires inverted LLRs
    
    command = [
        'ldpc-toolbox', 'decode',
        code["alist"],
        decoder,
        str(max_iterations),
        #'--output', '/dev/null',  # We don't need the output
        '-v' # Verbose - get debug data on stderr
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_data, stderr_data = process.communicate(input=data_f32.tobytes())
    process.wait()
    print(stdout_data)

    for line in stderr_data.splitlines():
        print(line)        

# Decoding
INFILE = "/tmp/infile"
def decode(data, code, max_iterations=100, decoder="Phif64"):
    """
    Returns the number of successfully decoded bits
    """
    with open(INFILE, "wb") as f: # NB needs removing to parallelize
        d = data.astype(np.float32) * -1 # Hack: somehow the decoder we're using requires inverted LLRs
        d.tofile(f)

    with NamedTemporaryFile() as out_file:
        tb = ldpc_ng(
            alist = code["alist"],
            k = code["k"],
            n = code["n"],
            max_iterations = max_iterations,
            decoder = decoder,
            in_file = INFILE,
            out_file = out_file.name,
        )

        tb.start()
        tb.wait()
        
        return os.stat(out_file.name).st_size

def generate_pulsed_jammer(JSR, pulse_length, frame_length, n_frames, gaussian=False, sync=False, rot=0, interleaver=None, offset=None):
    pulse_rate = pulse_length / frame_length
    peak_amplitude = np.sqrt(JSR / pulse_rate)
    gaussian_power = (JSR/pulse_rate)
    array = np.zeros((n_frames, frame_length), dtype=np.complex128)
    for i in range(n_frames):
        # Choose a random starting point for the pulse
        if offset is None:
            start_index = random.randint(0, frame_length - 1)
        else:
            start_index = offset
        circular_index = (start_index + pulse_length) % frame_length
        if circular_index < start_index:
            array[i, start_index:] = gaussian_noise(array.shape[1] - start_index, gaussian_power) if gaussian else peak_amplitude
            array[i, :circular_index] = gaussian_noise(circular_index, gaussian_power) if gaussian else peak_amplitude
        elif circular_index == start_index:
            if pulse_rate == 1:
                array[i, :] = gaussian_noise(array.shape[1], gaussian_power) if gaussian else peak_amplitude # XXX: added i, - is this correct?
        else:
            array[i, start_index:circular_index] = gaussian_noise(circular_index - start_index, gaussian_power) if gaussian else peak_amplitude

    jamming_signal = array.ravel().astype(np.csingle)

    # Apply a rotation if needed
    if sync:
        angles = np.zeros(len(jamming_signal))
    else:
        angles = np.random.uniform(0, 2*np.pi, len(jamming_signal))

    angles += rot
    jamming_signal *= np.exp(1j * angles).astype(np.csingle)

    # Apply interleaving if needed
    if interleaver == "ideal":
        # Shuffle at random
        np.random.shuffle(jamming_signal)
    elif interleaver == "row-column":
        original_length = len(jamming_signal)
        square_size = int(np.ceil(np.sqrt(original_length)))
        
        # Pad out the jamming signal
        padded_length = square_size ** 2
        if len(jamming_signal) < padded_length:
            jamming_signal = np.pad(jamming_signal, (0, padded_length - original_length), mode='constant')

        # Reshape into a 2D array (rows)
        reshaped_signal = jamming_signal.reshape((square_size, square_size))

        # Interleave by reading out columns
        interleaved_signal = reshaped_signal.T.flatten()  # Transpose and flatten to get column-wise order

        # Trim back down to original size
        # (Jamming symbols coinciding with these null regions are zeroed out)
        jamming_signal = interleaved_signal[:original_length]
    
    return jamming_signal

@dataclass
class JammerParameters():
    JSR: float
    JSR_dB: float = field(init=False)
    JSR_measured: float
    JSR_measured_dB: float = field(init=False)
    pulse_length: int
    gaussian: bool
    sync: bool
    full_knowledge: bool
    rot: float = 0

    def __post_init__(self):
        self.JSR_dB = to_dB(self.JSR)
        self.JSR_measured_dB = to_dB(self.JSR_measured)

@dataclass
class ProtocolParameters():
    modulation: str
    interleaver: str # "none", "ideal", "row-column" # Interleaving at this level not expected to work well
    code: str

@dataclass
class ReceiverParameters():
    N0: float
    N0_dB: float = field(init=False)
    decoder: str
    max_iterations: int

    def __post_init__(self):
        self.N0_dB = to_dB(self.N0)

def run_experiment(victim_symbols, jamming_symbols, N0, N0_measured, f_demod, f_decod):
    assert(len(victim_symbols) == len(jamming_symbols))
    n = len(victim_symbols)
    signal = victim_symbols + jamming_symbols + gaussian_noise(n, N0)
    
    noise_sigma = np.sqrt(N0_measured*0.5)
    llrs = f_demod(signal, noise_sigma)
    return f_decod(llrs)

def eval_ldpc(protocol: ProtocolParameters, receiver: ReceiverParameters, jammer: JammerParameters, n_codeblocks):
    """
    Evaluate the error rate of an LDPC frame
    """

    # This should be replaced with a nice data structure eventually
    if protocol.modulation == "BPSK":
        f_mod = mod_bpsk
        f_demod = demod_bpsk
        frame_length = protocol.code["n"]
    elif protocol.modulation == "QPSK":
        f_mod = mod_qpsk
        f_demod = demod_qpsk
        frame_length = protocol.code["n"]//2
    else:
        raise NotImplementedError()


    f_decod = partial(decode, code=protocol.code, max_iterations=receiver.max_iterations, decoder=receiver.decoder)
    f_cod = partial(encode, code=protocol.code, random_frames=not jammer.full_knowledge)

    jammer_symbols = generate_pulsed_jammer(
        JSR = jammer.JSR,
        pulse_length = jammer.pulse_length,
        frame_length = frame_length,
        n_frames=n_codeblocks,
        gaussian=jammer.gaussian,
        sync=jammer.sync,
        rot=jammer.rot,
        interleaver=protocol.interleaver
    )

    victim_symbols = f_mod(f_cod(n_frames=n_codeblocks))

    N0_measured = receiver.N0 + jammer.JSR_measured

    correct_bits = run_experiment(victim_symbols, jammer_symbols, receiver.N0, N0_measured, f_demod, f_decod)
    total_bits = n_codeblocks * protocol.code["k"]

    return {
        **asdict(protocol),
        **asdict(receiver),
        **asdict(jammer),
        **protocol.code,
        "N0_measured": N0_measured,
        "N0_measured_dB": to_dB(N0_measured),
        "n_codeblocks": n_codeblocks,
        "correct_bits": correct_bits,
        "total_bits": total_bits,
        "error_rate": 1 - (correct_bits / total_bits),
        "pulse_rate": jammer.pulse_length / frame_length
    }

# Later on, for the desynced jammer, we'll want to iterate over peak power rather than JSR measured
def grid_search(n_codeblocks, JSR_range, JSR_measured_range, pulse_length_range, gaussian, sync, full_knowledge, rot, protocol: ProtocolParameters, receiver: ReceiverParameters):
    """
    Search over a grid of params
    """

    res = []
    for pulse_length in pulse_length_range:
        for (JSR, JSR_measured) in tqdm(list(zip(JSR_range, JSR_measured_range))):
            jammer = JammerParameters(
                JSR = JSR,
                JSR_measured = JSR_measured,
                pulse_length = pulse_length,
                gaussian = gaussian,
                sync = sync,
                full_knowledge = full_knowledge,
                rot = rot
            )
            r = eval_ldpc(protocol, receiver, jammer, n_codeblocks)
            res.append(r)
            if r["error_rate"] == 1.0:
                break

    return res

if __name__ == "__main__":
    # Perform a test grid search

    code = codes[0]

    protocol = ProtocolParameters(
        modulation = "BPSK",
        interleaver = "none",
        code = code
    )

    receiver = ReceiverParameters(
        N0 = from_dB(-15),
        decoder = "Aminstarf32",
        max_iterations = 10
    )

    df = pd.DataFrame(
        grid_search(
            n_codeblocks = 100,
            JSR_range = np.arange(-2.0, 4.5, 0.1),
            JSR_measured_range = np.arange(-2, 4.5, 0.1), # accurate measurement
            pulse_length_range = np.linspace(0, code["n"], 5, dtype=int),
            gaussian = False,
            sync = False,
            full_knowledge = False,
            rot = 0,
            protocol=protocol,
            receiver=receiver
        )
    )
    print(df)
