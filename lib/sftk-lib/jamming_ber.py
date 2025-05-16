import numpy as np
import matplotlib.pyplot as plt
#import commpy

def measure_power(iqs):
    return np.average(iqs.real**2 + iqs.imag**2)

def gaussian_noise(n, N0, dtype=None):
    sigma = np.sqrt(N0/2)
    noise = np.random.normal(0,sigma,n) + 1j * np.random.normal(0,sigma,n)
    if dtype is not None:
        return noise.astype(dtype)
    else:
        return noise

def measure_constellation_Eb(constellation_iq):
    bit_per_symbol = np.log2(len(constellation_iq))
    energy_per_symbol = measure_power(constellation_iq)
    #energy_per_bit = energy_per_symbol / bit_per_symbol
    return energy_per_symbol

def norm_constellation(constellation_iq):
    factor = np.sqrt(measure_constellation_Eb(constellation_iq))
    constellation_iq /= factor

    return constellation_iq, factor

def plot_constellation(cons):
    constellation = cons[0]
    bits = cons[1]
    plt.scatter(np.real(constellation), np.imag(constellation), c=range(len(constellation)))
    n = max(map(lambda x: len(bin(x)[2:]), bits))
    for c, b in zip(constellation, bits):
        plt.text(c.real, c.imag, format(b, f"0{n}b"))
    ax = plt.gca()
    ax.axis('equal')

def generate_psk_constellation(M):
    constellation, _ = norm_constellation(np.exp(1j * np.linspace(0, 2*np.pi, M, endpoint=False)))
    symbol_ids = np.array(range(M))
    symbol_bits = symbol_ids ^ (symbol_ids >> 1) # gray coding
    return constellation, symbol_bits

def generate_rrc_impulse_response(sample_rate=100000, samples_per_symbol=1000, Beta=0.35, N=10):
    symbol_rate = sample_rate/samples_per_symbol
    Ts = 1/symbol_rate
    #Ts = samples_per_symbol/sample_rate

    def get_impulse(x):
        if x == 0:
            return (1/Ts)*(1 + Beta*((4/np.pi - 1)))
        elif x == Ts/(4*Beta) or x == -Ts/(4*Beta):
            return (Beta/(Ts*np.sqrt(2)))*((1 + 2/np.pi)*((np.sin(np.pi/(4*Beta))) + (1-(2/np.pi)*np.cos(np.pi/(4*Beta)))))
        else:
            return (1/Ts) * ((np.sin((np.pi*x/Ts)*(1-Beta)) + (4*Beta)*(x/Ts) * np.cos((np.pi*x/Ts)*(1+Beta))) / (((np.pi*x)/Ts)*(1-((4*Beta*x)/Ts)**2)))

    xs = np.linspace(-N*Ts, N*Ts, 2*N*samples_per_symbol)
    impulse_response = np.vectorize(get_impulse)(xs)

    #plt.plot(xs, impulse_response)
    #plt.plot(np.fft.fftfreq(len(xs), 1/sample_rate), np.abs(np.fft.fft(impulse_response)))
    #plt.show()
    return impulse_response/np.sum(impulse_response)

def get_victim_iqs(iqs, impulse_response, samples_per_symbol, offset=0):
    # Convolve the symbols, sample at the correct time, return the IQs

    # Generate the dirac deltas
    input = np.zeros(int((len(iqs)+0.5) * samples_per_symbol), dtype=np.csingle)
    input[int(samples_per_symbol/2)::samples_per_symbol] = iqs

    # Convolve the filter
    xs = np.convolve(input, impulse_response, mode="valid")
    xs = xs / np.sqrt(np.mean(np.abs(xs)**2))

    # Sample the signal
    xs_samples = xs[np.nonzero(input)+offset]
    return xs_samples

def get_attacker_iqs(iqs, impulse_response, samples_per_symbol):
    # Generate the square wave
    input = np.repeat(iqs, samples_per_symbol)

    # Convolve the filter
    xs = np.convolve(input, impulse_response, mode="valid")
    xs = xs / np.sqrt(np.mean(np.abs(xs)**2))

    # Sample the signal
    n = int((len(xs)-1)/samples_per_symbol)
    choices = np.random.randint(0, samples_per_symbol, n)
    choices += np.arange(0, int(len(xs)-1)-samples_per_symbol, samples_per_symbol)
    out_iqs = xs[choices]

    return out_iqs


def get_filtered_iqs(iqs, impulse_response, samples_per_symbol):
    input = np.zeros(int((len(iqs)+0.5) * samples_per_symbol), dtype=np.csingle)
    input[int(samples_per_symbol/2)::samples_per_symbol] = iqs
 
    xs = np.convolve(input, impulse_response, mode="valid") # This throws away a few samples, but we don't pay this much mind
    xs = xs / np.sqrt(np.mean(np.abs(xs)**2))

    #print("'''")
    #print(len(iqs))
    #print(len(xs))

    #n2 = int((len(iqs) - int((len(xs)-1)/samples_per_symbol) )/2)
    #print(n2)
    #input_trunc = input[n2*samples_per_symbol:-n2*samples_per_symbol]

    #plt.plot(input_trunc.real)
    #plt.plot(input_trunc.imag)

    #plt.plot(xs.real)
    #plt.plot(xs.imag)

    #print("###")
    #print(np.sqrt(np.mean(np.abs(xs)**2)))

    n = int((len(xs)-1)/samples_per_symbol)
    choices = np.random.randint(0, samples_per_symbol, n)
    choices += np.arange(0, int(len(xs)-1)-samples_per_symbol, samples_per_symbol)
    out_iqs = xs[choices]

    return out_iqs

# Ls - list of bit labels per ring
# Rs - list of radii of each ring
# As - list of angular offsets of each ring
def generate_apsk_constellation(Ls, Rs, As):
    constellation = []
    labels = []

    for L, R, A in zip(Ls, Rs, As):
        M = len(L)
        cons = generate_psk_constellation(M)[0]

        # Scale and rotate the constellation
        cons = map(lambda x: x * (np.cos(A) + np.sin(A)*1j) * R, cons)

        # Apply the labels
        constellation += cons
        labels += L

    (cons, factor) = norm_constellation(np.array(constellation))
    return cons, np.array(labels), factor

def generate_qam_constellation(M):
    sqrtM = int(np.sqrt(M))
    bits_per_side = int(np.log2(sqrtM))
    sides = np.linspace(-1, 1, sqrtM, endpoint=True)
    constellation_i, constellation_q = np.meshgrid(sides, sides)
    constellation, factor = norm_constellation((constellation_i + 1j*constellation_q).flatten())

    symbol_ids = np.array(range(sqrtM))
    symbol_bits = symbol_ids ^ (symbol_ids >> 1)
    symbol_bits_r, symbol_bits_c = np.meshgrid(symbol_bits, symbol_bits)
    symbol_bits_full = (symbol_bits_r.flatten() << bits_per_side) + symbol_bits_c.flatten()

    return constellation, symbol_bits_full, (-1/factor, 2/(sqrtM-1)/factor)

psk_constellations = {
    2: generate_psk_constellation(2),
    4: generate_psk_constellation(4),
    8: generate_psk_constellation(8),
    16: generate_psk_constellation(16),
}

qam_constellations = {
    4: generate_qam_constellation(4),
    16: generate_qam_constellation(16),
    64: generate_qam_constellation(64),
    256: generate_qam_constellation(256),
}


def decode_generic(const_iqs, iqs):
    def decode_single(iq):
        return np.argmin(np.abs(const_iqs - iq))

    return np.vectorize(decode_single)(iqs)

#    res = []
#    for i in iqs:
#        res.append(np.argmin(np.abs(const_iqs - i)))
#    return res

# TODO: refactor these to output bit streams rather than constellation points

#def demodulate_generic_soft(const_iqs, iqs, sigma=1):
    # Find the nearest point to the received sample with a 0, and with a 1, at that bit position
    # Then compute -1/(sigma**2) * ()
    # https://uk.mathworks.com/help/comm/ug/digital-baseband-modulation.html#bu_zzah-2

    # Find the 

# A more efficient decoder could potentially be written based on the below
# def decode_apsk_constellation(labels, Rs, As, iqs):
#     (_, _, factor) = generate_apsk_constellation(labels, Rs, As)
#     rings = list(map(len, labels))
# 
#     # Scale the constellation radii
#     Rs = np.array(Rs) / factor
# 
#     # First determine the ring we're in
#     midpoints = np.array(list(map(np.mean, zip(Rs, Rs[1:]))))
#     print(midpoints)
#     iq_rings = np.vectorize(lambda x: np.sum(np.abs(x) >= midpoints))(iqs)
# 
#     return iq_rings
# 
#     print(f"iq_rings: {len(iq_rings)}")
#     print(f"iqs: {len(iqs)}")
# 
#     # Then determine the angular offset
#     return list(map(
#         lambda x:
#             decode_psk_constellation(rings[x[0]], x[1], -As[x[0]]) +
#             sum(rings[:x[0]]),
#         zip(iq_rings, iqs)))

def decode_psk_constellation(M, iq, offset=0):
    return np.mod(np.rint((np.angle(iq) + offset) * M / (2*np.pi)), M).astype(int)

def decode_qam_constellation(M, iq):
    Msqrt = np.sqrt(M)
    _, _, gap = qam_constellations[M]
    column = np.clip(np.rint((np.real(iq) - gap[0]) / gap[1]), 0, Msqrt - 1)
    row = np.clip(np.rint((np.imag(iq) - gap[0]) / gap[1]), 0, Msqrt - 1)
    return (row * Msqrt + column).astype(int)

def random_iq(n, range):
    i = 2*range*np.random.random(n) - range
    q = 2*range*np.random.random(n) - range
    return i+1j*q

def plot_decode(M, iq, decoder):
    decoded = decoder(M, iq)
    plt.scatter(np.real(iq), np.imag(iq), c=decoded, s=1)

def generate_bit_accelerator_table(M):
    return np.array([bin(i).count("1") for i in range(M)])

BIT_ACCELERATOR_TABLE = generate_bit_accelerator_table(1024)

def measure_bit_errors(b1, b2):
    return np.sum(BIT_ACCELERATOR_TABLE[b1 ^ b2])

# TODO: implement RRC filter like this guy: https://github.com/veeresht/CommPy/issues/39
def filtering_scratchpad():
    samples_per_symbol = 4
    sample_rate = 100000
    symbol_rate = sample_rate / samples_per_symbol

    pos, taps = commpy.rrcosfilter(int(samples_per_symbol * 8), 0.35, 1.0/symbol_rate, sample_rate)

# TODO: add offsets due to wave desync
# https://commpy.readthedocs.io/en/latest/filters.html
def attack_generic(constellation, M, n, EaEvdB, N0EvdB, align=False, filter=None, attacker_strategy="symbol", phase_offset=0):
    #Ev is assumed to be one
    n_bits = int(np.log2(len(constellation[0])))
    victim_samples = np.random.randint(0, len(constellation[0]), n)
    total_iq = constellation[0][victim_samples]
    n_buffer_samples = 10 # Needed because the filter convolution will throw some away

    if EaEvdB is not None:
        if attacker_strategy == "symbol":
            attacker_samples = np.random.randint(0, 4, n+n_buffer_samples)
            attacker_voltage_scale = 10**(EaEvdB/20)
            attacker_iq = psk_constellations[4][0][attacker_samples] * attacker_voltage_scale
            #attacker_iq = [attacker_voltage_scale]*(n+n_buffer_samples) # Generate samples more cheaply if we're immediately going to phase misalign

            if filter is not None:
                attacker_iq = np.array(get_filtered_iqs(attacker_iq, filter, samples_per_symbol=1000)) * attacker_voltage_scale # TODO: fix hardcoding samples_per_symbol

        else:
            raise Exception(f"Unknown attacker strategy {attacker_strategy}")


        if phase_offset != 0:
            attacker_iq *= np.exp(1j*phase_offset)

        if not align:
            attacker_phases = (2*np.pi)*np.random.random(len(attacker_iq))
            attacker_iq *= np.exp(1j*attacker_phases)


        total_iq += attacker_iq[:n]

    if N0EvdB is not None:
        N0 = 10**(N0EvdB/10) # is this right? not 20? Maybe, since N0 is noise power
        noise_iq = gaussian_noise(n, N0)

        total_iq += noise_iq

    decoded_samples = decode_generic(constellation[0], total_iq)

    victim_bits = constellation[1][victim_samples]
    decoded_bits = constellation[1][decoded_samples]

    bit_errors = measure_bit_errors(victim_bits, decoded_bits)
    total_bits = n_bits * n
    return bit_errors / total_bits

def attack_psk(M, n, EaEvdB, N0EvdB, align = False):
    return attack_generic(psk_constellations[M], decode_psk_constellation, M, n, EaEvdB, N0EvdB, align)

def attack_qam(M, n, EaEvdB, N0EvdB, align = False):
    return attack_generic(qam_constellations[M], decode_qam_constellation, M, n, EaEvdB, N0EvdB, align)
