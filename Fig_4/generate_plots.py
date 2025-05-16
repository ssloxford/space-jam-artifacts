#!/usr/bin/python3

# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation version 3.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

# Top: LDPC simulation performance comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import re
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import norm
import json

from lib.modulation.dvb_s2 import BPSK, QPSK, PSK_8, APSK_8_rate_100_180, APSK_16_rate_4_5, APSK_32_rate_4_5
from lib.modulation import QAM
from lib.misc import from_dB, to_dB, plot
from lib.codeword_aware import generate_distance_distributions, optimal_error_rates

os.makedirs(os.getcwd() + "/plot/Fig_4/", exist_ok=True)
os.makedirs(os.getcwd() + "/out/", exist_ok=True)

def get_color(id, count):
    def lerp(a, b, x):
        return a * (1-x) + b * x
    x = (id+1) / count
    return ((lerp(1, 0, x)),(lerp(1, (33/255), x)),(lerp(1, (71/255), x)))

@plot
def lineplot(df, df_opt, opt_threshold=None, legend=True, ax=None):
    golden = (1 + 5 ** 0.5) / 2
    height = 3
    if ax==None:
        fig = plt.figure(figsize=(height * golden, height), frameon=False)
        ax = fig.add_subplot(1, 1, 1)

    color_map = {
        'full_knowledge': 'blue',
        'desync_symbol': 'green',
        'sync_symbol': 'purple',
        'gaussian': 'orange'
    }
    
    color_map = {
        'lub': get_color(2, 7),
        'full_knowledge': get_color(3, 7),
        'desync_symbol': get_color(4, 7),
        'sync_symbol': get_color(5, 7),
        'gaussian': get_color(6, 7),
    }
    
    label_map = {
        'full_knowledge': "Codeword-aware",
        'desync_symbol': "Desynchronized",
        'sync_symbol': "Synchronized",
        'gaussian': "Noise (AWGN)"
    }

    _df = df.copy()

    # Plot the lines
    sns.lineplot(df_opt, x="JSR_dB", y="error_rate", color=color_map["lub"], label="Full knowledge")
    
    labels = _df['label'].unique()
    for i, label in enumerate(labels):
        # Find the best parameters to maximize jamming under interleaving
        __df = _df[_df["label"] == label]
        idx = __df.groupby(['label', 'JSR'])['error_rate'].idxmax()
        result_df = __df.loc[idx].reset_index(drop=True)

        if label == "desync_gaussian":
            continue # We consider Gaussian separately below
        if label == "full_knowledge":
            continue
        sns.lineplot(result_df[result_df["label"] == label], x="JSR_dB", y="error_rate", color=color_map[label], label=label_map[label])

    # Find the continuous Gaussian jammer
    gaussian = _df[(_df["pulse_rate"] == 1) & (_df["label"] == "desync_gaussian")]

    sns.lineplot(gaussian, x="JSR_dB", y="error_rate", color=color_map["gaussian"], label=label_map["gaussian"])
    
    if opt_threshold:
        ax.axvline(to_dB(opt_threshold), ymin=0, ymax=1, color=color_map["lub"], linestyle="dotted")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('JSR [dB]')
    ax.set_ylabel('Error Rate')
    
    # Add the dotted lines into the legend
    lines = []
    lines.append(plt.plot([np.NaN], linestyle="dotted", color="black")[0])    
    
    labels = [r"Zero noise, $N_0 = 0$"]
    
    current_handles, current_labels = plt.gca().get_legend_handles_labels()

    # Combine the existing handles and labels with the new ones
    new_handles = current_handles + lines
    new_labels = current_labels + labels

    if legend:
        # Update the legend with the new handles and labels
        plt.legend(new_handles, new_labels)

        plt.legend(new_handles, new_labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.3))
    else:
        ax.get_legend().remove()

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Create the dataframes of the optimal bound

def get_threshold(rho_min, n, d_H):
    return rho_min * (d_H/n)

constellations = {
    "BPSK": BPSK,
    "QPSK": QPSK,
    "8-PSK": PSK_8,
    "8-APSK": APSK_8_rate_100_180,
}

N0 = from_dB(-15)

# (128,64) TC code
d_H_tc = 14
n_tc = 128
k_tc = 64

Pjs = from_dB(np.arange(-15, -5, 0.01))
df_tc_opt_bpsk = pd.DataFrame(optimal_error_rates(Pjs, BPSK, n_tc, d_H_tc, N0))
df_tc_opt_qpsk = pd.DataFrame(optimal_error_rates(Pjs, QPSK, n_tc, d_H_tc, N0))

# Read in the data
df_tc = pd.read_pickle("./data/df_i.pkl")
df_tc_qpsk = pd.read_pickle("./data/df_tc_qpsk.pkl")

# Plot
d_H_tc = 14
n_tc = 128
k_tc = 64

# TODO: separate out legend
lineplot(df_tc, df_tc_opt_bpsk, opt_threshold=get_threshold(1.0, n_tc, d_H_tc),
         output_pdf="Fig_4/bpsk.pdf", show=False)

lineplot(df_tc_qpsk, df_tc_opt_qpsk, opt_threshold=get_threshold(0.5, n_tc, d_H_tc),
         output_pdf="Fig_4/qpsk.pdf", legend=False, show=False)


# Bottom

# Compute the sync headers

## Headers
headers = {
    "TM (short)": int('1acffc1d', 16),
    "TM": int('034776C7272895B0', 16),
    "TC (short)": int('EB90', 16),
    "TC": int('034776C7272895B0', 16),
    "DVB-S2": int('18D2E82', 16),
}

# Modulate each header with each relevant modulation scheme
modulated_headers = {
    "TC": BPSK.modulate(headers["TC"]),
    "TC (short)": BPSK.modulate(headers["TC (short)"]),
    "TM": QPSK.modulate(headers["TM"]),
    "TM (short)": QPSK.modulate(headers["TM (short)"]),
    "DVB-S2": BPSK.modulate(headers["DVB-S2"])
}

modulated_headers_2 = {
    "QPSK": QPSK.modulate(headers["TM"]),
    "8-PSK": PSK_8.modulate(headers["TM"]),
    "16-APSK": APSK_16_rate_4_5.modulate(headers["TM"]),
    "32-APSK": APSK_32_rate_4_5.modulate(headers["TM"]),
    "64-QAM": QAM(64).modulate(headers["TM"])
}

modulated_headers_3 = {
    "TC": BPSK.modulate(headers["TC"]),
    "TM": QPSK.modulate(headers["TM"]),
    "DVB-S2": BPSK.modulate(headers["DVB-S2"])
}

header_distance_vectors = {
    k: np.abs(v) for (k,v) in modulated_headers.items()
}

header_distance_vectors_2 = {
    k: np.abs(v) for (k,v) in modulated_headers_2.items()
}

header_distance_vectors_3 = {
    k: np.abs(v) for (k,v) in modulated_headers_3.items()
}

header_distances = {
    k: np.sqrt(np.sum(np.abs(v)**2)) for (k,v) in modulated_headers.items()
}

header_distances_2 = {
    k: np.sqrt(np.sum(np.abs(v)**2)) for (k,v) in modulated_headers_2.items()
}

header_distances_3 = {
    k: np.sqrt(np.sum(np.abs(v)**2)) for (k,v) in modulated_headers_3.items()
}

# For each distance in the distribution, 

def gaussian_error_rate(N0, threshold):
    # Calculate the standard deviation from the variance
    std_dev = np.sqrt(N0 / 2)
    
    # Calculate the CDF for the given thresholds
    cdf_values = norm.cdf(threshold, loc=0, scale=std_dev)
    
    # Calculate the probability of exceeding the threshold
    error_rate = 1 - cdf_values
    
    return error_rate

def gaussian_error_rate_overall(Pj, N0, threshold, distance_vector):
    Pj_per_symbol = Pj/len(distance_vector)
    return gaussian_error_rate(N0 + Pj_per_symbol, threshold)

def desync_error_rate(Pj, N0, threshold, distance_vector, n_trials):
    l = []
    for _Pj in Pj:
        Pj_per_symbol = _Pj/len(distance_vector)
        distance_threshold = np.sqrt(np.sum(np.abs(distance_vector**2)))
        norm_distance = distance_vector / distance_threshold
        #print(f"distance_threshold: {distance_threshold}")

        shape = (n_trials, len(distance_vector))
        symbs = np.full(shape, Pj_per_symbol).astype(np.csingle)
        symbs *= np.exp(1j * np.random.uniform(0, 2*np.pi, shape))
        distances = np.sum(symbs.real * norm_distance, axis=1)

        #print(distances)

        std_dev = np.sqrt(N0 / 2)
        cdf_values = 1-norm.cdf(threshold, loc=distances, scale=std_dev)

        l.append(cdf_values.mean())
    return np.array(l)
            
def optimal_error_rate(Pj, N0, threshold):
    std_dev = np.sqrt(N0 / 2)
    ts = threshold - np.sqrt(Pj)
    cdf_values = norm.cdf(ts, loc=0, scale=std_dev)
    error_rate = 1 - cdf_values
    return error_rate

## Parameters
threshold = 0
header_distances

# Parameters
N0 = from_dB(-15)
t = 1
n_trials = 1000 # 10000

Pj_range_dB = np.arange(-4, 20, 0.1) # 0.01
Pj_range = from_dB(Pj_range_dB)

@plot
def f(hdv, ncol=2):
    fig, ax = plt.subplots(1, figsize=(5,2.7))
    for i, (name, d_v) in enumerate(hdv.items()):
        color = get_color(i+2, len(hdv.items())+2)
        
        print(name)
        n_symbs = len(d_v)
        Pjs = Pj_range * n_symbs

        distance_threshold = np.sqrt(np.sum(np.abs(d_v**2)))
        threshold = distance_threshold * t
        print(f"threshold: {threshold}")

        # Calculate the CDFs
        optimal = optimal_error_rate(Pjs, N0, threshold)
        desync = desync_error_rate(Pjs, N0, threshold, d_v, n_trials)
        gaussian = gaussian_error_rate_overall(Pjs, N0, threshold, d_v)

        sns.lineplot(x=Pj_range_dB, y=gaussian, color=color, linestyle="dotted", ax=ax)
        sns.lineplot(x=Pj_range_dB, y=desync, color=color, linestyle="dashed", ax=ax)
        sns.lineplot(x=Pj_range_dB, y=optimal, color=color, linestyle="solid", label=name, ax=ax)

    # Add to legend
    lines = []
    lines.append(plt.plot([np.NaN], linestyle="-", color="black", alpha=0.0, label=" ")[0])
    lines.append(plt.plot([np.NaN], linestyle="-", color="black", label="Synchronized")[0])
    lines.append(plt.plot([np.NaN], linestyle="dashed", color="black", label="Desynchronized")[0])
    lines.append(plt.plot([np.NaN], linestyle="dotted", color="black", label="Gaussian")[0])
        
    
        
    plt.legend(loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1.4))
    plt.ylabel("Preamble Error Rate")
    plt.xlabel("JSR [dB]")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
f(hdv = header_distance_vectors, ncol=3, output_pdf="Fig_4/standards.pdf", show=False)
f(hdv = header_distance_vectors_2, ncol=3, output_pdf="Fig_4/constellations.pdf", show=False)

# Calculate the values for Table 2
hdv = header_distance_vectors_3

# Size of the sync header relative to the size of the overall frame, in dB
total_size = {
    "TC": 448,
    "TM": 5184,
    "DVB-S2": 32890,
}

preamble_size = {
    "TC": 64,
    "TM": 64,
    "DVB-S2": 26,
}

half_dB = 10*np.log10(49/99)

# Parameters
N0 = from_dB(-15)
n_trials = 1000 # 10000

Pj_range_dB = np.arange(-10, 100, 0.01)
Pj_range = from_dB(Pj_range_dB)

for i, (name, d_v) in enumerate(hdv.items()):
    threshold = np.sqrt(np.sum(np.abs(d_v**2)))
    
    optimal = optimal_error_rate(Pj_range, N0, threshold)
    desync = desync_error_rate(Pj_range, N0, threshold, d_v, n_trials)
    gaussian = gaussian_error_rate_overall(Pj_range, N0, threshold, d_v)
    
    # Find the 50% and 100% thresholds for each
    sync_50 = Pj_range_dB[np.argmax(optimal > 0.49)]
    desync_50 = Pj_range_dB[np.argmax(desync > 0.49)]
    gaussian_50 = Pj_range_dB[np.argmax(gaussian > 0.49)]
    sync_100 = Pj_range_dB[np.argmax(optimal > 0.99)]
    desync_100 = Pj_range_dB[np.argmax(desync > 0.99)]
    gaussian_100 = Pj_range_dB[np.argmax(gaussian > 0.99)]
    
    preamble = preamble_size[name]
    total = total_size[name]

    data = {
            "0_gaussian": {
                "100%": None,
                "50%": gaussian_50 + to_dB(1/preamble),
            },
            "1_interleaved": {
                "100%": None,
                "50%": desync_50 + to_dB(1/preamble),
            },
            "2_deinterleaved": {
                "100%": None,
                "50%": desync_50 + to_dB(1/preamble),
            },
            "3_frame_sync": {
                "100%": None,
                "50%": desync_50 + to_dB(1/total),
            },
            "4_coherent": {
                "100%": sync_100 + to_dB(1/total),
                "50%": sync_100 + to_dB(1/total) + half_dB,
            },
            "5_full_knowledge": {
                "100%": sync_100 + to_dB(1/total),
                "50%": sync_100 + to_dB(1/total) + half_dB,
            }
    }

    with open(f"./out/preamble_{name}.json", "w") as f:
        json.dump(data, f, indent=4)  # Use indent for pretty printing
