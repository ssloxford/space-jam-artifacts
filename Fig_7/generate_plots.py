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

# Generate Figure 7

from ldpc_ber import grid_search, codes, ProtocolParameters, ReceiverParameters
from lib.misc import from_dB, to_dB, plot
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

os.makedirs(os.getcwd() + "/plot/Fig_7/", exist_ok=True)

# Define plotting helpers

from matplotlib.colors import LinearSegmentedColormap
golden = (1 + 5 ** 0.5) / 2
height = 3

white_to_color = LinearSegmentedColormap.from_list('custom_cmap', ["#002147", "white"])  # RGB color with blue channel set to 1

@plot
def plot_dataframe(_df, label, N=1, cbar=True):
    fig = plt.figure(figsize=(height * golden, height), frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    
    __df = _df.copy()
    __df["error_rate"] = 1 - (1 - __df["error_rate"])**N
    data = __df.pivot(index='pulse_rate', columns='JSR_dB', values='error_rate')
    
    sns.heatmap(data.round(1), annot=data.round(1), cmap=white_to_color, ax=ax, cbar=cbar, annot_kws={"fontsize":8})
    #ax.set_title(label)
    ax.set_xlabel('JSR [dB]')
    ax.set_ylabel('Pulse Rate')
    
    # Format the labels correctly
    fmt = '{:0.2f}'
    xticklabels = []
    for item in ax.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels += [item]
    yticklabels = []
    for item in ax.get_yticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        yticklabels += [item]

    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

@plot
def plot_colorbar():
    # Create a new figure for the colorbar
    fig, ax = plt.subplots(figsize=(height/20, height))  # Adjust the size as needed
    norm = plt.Normalize(0, 1)  # Normalize the colorbar values
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=white_to_color), cax=ax)  # Create the colorbar
    cbar.set_label('Codeword Error Rate')  # Set the label for the colorbar
    
    # Remove the border around the colorbar
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Set the edge color of the colorbar to be transparent
    cbar.outline.set_visible(False)
    
def get_color(id, count):
    def lerp(a, b, x):
        return a * (1-x) + b * x
    x = (id+1) / count
    return ((lerp(1, 0, x)),(lerp(1, (33/255), x)),(lerp(1, (71/255), x)))

def set_cycler(count, back = False):
    cols = [get_color(i, count) for i in range(count)]
    if back:
        cols = cols[::-1]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cols)
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

@plot
def lineplot(df, N, legend=True):
    height = 3
    fig = plt.figure(figsize=(height * golden, height), frameon=False)
    ax = fig.add_subplot(1, 1, 1)

    color_map = {
        'full_knowledge': 'blue',
        'desync_symbol': 'green',
        'sync_symbol': 'purple',
        'gaussian': 'orange'
    }
    
    color_map = {
        'full_knowledge': get_color(2, 6),
        'desync_symbol': get_color(3, 6),
        'sync_symbol': get_color(4, 6),
        'gaussian': get_color(5, 6)
    }
    
    label_map = {
        'full_knowledge': "Full knowledge",
        'desync_symbol': "Non-coherent",
        'sync_symbol': "Coherent",
        'gaussian': "Gaussian"
    }

    _df = df.copy()
    _df["error_rate_interleaved"] = 1 - (1 - _df["error_rate"]) ** N

    # Plot the lines
    labels = _df['label'].unique()
    for i, label in enumerate(labels):
        # Find the best parameters to maximize jamming under interleaving
        __df = _df[_df["label"] == label]
        idx = __df.groupby(['label', 'JSR'])['error_rate'].idxmax()
        result_df = __df.loc[idx].reset_index(drop=True)

        if label == "desync_gaussian":
            continue # We consider Gaussian separately below
        sns.lineplot(result_df[result_df["label"] == label], x="JSR_dB", y="error_rate_interleaved", color=color_map[label], label=label_map[label])
        sns.lineplot(result_df[result_df["label"] == label], x="JSR_dB", y="error_rate", linestyle="dotted", color=color_map[label])

    # Find the continuous Gaussian jammer
    gaussian = _df[(_df["pulse_rate"] == 1) & (_df["label"] == "desync_gaussian")]

    sns.lineplot(gaussian, x="JSR_dB", y="error_rate_interleaved", color=color_map["gaussian"], label=label_map["gaussian"])
    sns.lineplot(gaussian, x="JSR_dB", y="error_rate", linestyle="dotted", color=color_map["gaussian"])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('JSR [dB]')
    ax.set_ylabel('Error Rate')
    
    # Add the dotted lines into the legend
    lines = []
    lines.append(plt.plot([np.nan], linestyle="-", color="black")[0])    
    lines.append(plt.plot([np.nan], linestyle="dotted", color="black")[0])    
    
    labels = ["Frame Error Rate", "Codeword Error Rate"]
    
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

# Indexed according to `codes`
code_parameters = [
    {
        "limits": {
            ("desync", "gaussian"): (-3.0, 3.0),
            ("desync", "symbol"): (-4.0, 4.0),
            ("sync", "symbol"): (-6.0, 0.0),
            ("sync", "full-knowledge"): (-8.0, 0.0)
        },
        "max_iterations": 100
    },
    {
        "limits": {
            ("desync", "gaussian"): (-0.5, 4.0),
            ("desync", "symbol"): (-0.5, 4.0),
            ("sync", "symbol"): (-3.0, 3.0)
        },
        "max_iterations": 100
    },
    {
        "limits": {
            ("desync", "gaussian"): (-7, -3.0),
            ("desync", "symbol"): (-9, -5.0),
            ("sync", "symbol"): (-9, -5.0),
            ("sync", "full-knowledge"): (-11.0, -8.0)
        },
        "max_iterations": 100
    },
    {
        "limits": {
            ("desync", "gaussian"): (-8.0, -5.0),
            ("desync", "symbol"): (-6.0, -3.0),
            ("sync", "symbol"): (-7.5, -4.0)
        },
        "max_iterations": 25
    },
    {
        "limits": {
            ("desync", "gaussian"): (1.0, 4.0),
            ("desync", "symbol"): (1.0, 4.0),
            ("sync", "symbol"): (1.0, 4.0)
        },
        "max_iterations": 25
    }
]

def run(code, n_codeblocks, pulse_length_range, protocol, receiver, JSR_range=None, JSR_precision=10, just=None):
    df_tm = pd.DataFrame()

    print(code)
    print((np.linspace(*code["limits"][("sync", "symbol")], JSR_precision)))
    
    if just == "sync" or just is None:
        ### Sync symbol
        JSRs = JSR_range if JSR_range is not None else from_dB(np.linspace(*code["limits"][("sync", "symbol")], JSR_precision))
        res = grid_search(
            n_codeblocks=n_codeblocks,
            JSR_range = JSRs,
            JSR_measured_range = JSRs, # accurate measurement
            pulse_length_range = pulse_length_range,
            gaussian=False,
            sync=True,
            full_knowledge=False,
            rot=0,
            protocol=protocol,
            receiver=receiver
        )
        _df = pd.DataFrame(res)
        _df["label"] = "sync_symbol"
        df_tm = pd.concat([_df, df_tm])

    if just == "gaussian" or just is None:
        ### Desync Gaussian
        JSRs = JSR_range if JSR_range is not None else from_dB(np.linspace(*code["limits"][("desync", "gaussian")], JSR_precision))
        res = grid_search(
            n_codeblocks=n_codeblocks,
            JSR_range = JSRs,
            JSR_measured_range = JSRs, # accurate measurement
            pulse_length_range = pulse_length_range,
            gaussian=True,
            sync=False,
            full_knowledge=False,
            rot=0,
            protocol=protocol,
            receiver=receiver
        )
        _df = pd.DataFrame(res)
        _df["label"] = "desync_gaussian"
        df_tm = pd.concat([_df, df_tm])

    ### Desync symbol
    if just == "desync" or just is None:
        JSRs = JSR_range if JSR_range is not None else from_dB(np.linspace(*code["limits"][("desync", "symbol")], JSR_precision))
        res = grid_search(
            n_codeblocks=n_codeblocks,
            JSR_range = JSRs,
            JSR_measured_range = JSRs, # accurate measurement
            pulse_length_range = pulse_length_range,
            gaussian=False,
            sync=False,
            full_knowledge=False,
            rot=0,
            protocol=protocol,
            receiver=receiver
        )
        _df = pd.DataFrame(res)
        _df["label"] = "desync_symbol"
        df_tm = pd.concat([_df, df_tm])

    return df_tm


## Simulation parameters
N0 = from_dB(-15)
n_codeblocks = 100 # 2025-05-15: Was 10 before
JSR_precision = 1
n_pulse_lengths = 10
lower = -10
upper = 4

## CCSDS TC

code = codes[0]

protocol = ProtocolParameters(
    modulation = "BPSK",
    interleaver = "none",
    code = code
)

receiver = ReceiverParameters(
    N0 = N0,
    decoder = "Aminstarf32",
    max_iterations = code_parameters[0]["max_iterations"]
)

JSR_range = from_dB(np.arange(lower, upper, JSR_precision))
pulse_length_range = np.linspace(0, code["n"], n_pulse_lengths).astype(int)

df = run(code, n_codeblocks, pulse_length_range, protocol, receiver, JSR_range=JSR_range)

## Plot CCSDS TC

plot_colorbar(output_pdf="Fig_7/tc_cbar.pdf", show=False)

labels = df['label'].unique()
for i, label in enumerate(labels):
    _df = df[df["label"] == label]
    plot_dataframe(_df, label, cbar=False, output_pdf=f"Fig_7/tc_{label}.pdf", show=False)

## CCSDS TM

code = codes[2]

protocol = ProtocolParameters(
    modulation = "QPSK",
    interleaver = "none",
    code = code
)

receiver = ReceiverParameters(
    N0 = N0,
    decoder = "Aminstarf32",
    max_iterations = code_parameters[0]["max_iterations"]
)

JSR_range = from_dB(np.arange(lower, upper, JSR_precision))
pulse_length_range = np.linspace(0, code["n"]//2, n_pulse_lengths).astype(int)

df_tm = run(code, n_codeblocks, pulse_length_range, protocol, receiver, JSR_range=JSR_range)

plot_colorbar(output_pdf="Fig_7/tm_cbar.pdf", show=False)

labels = df_tm['label'].unique()
for i, label in enumerate(labels):
    _df = df_tm[df_tm["label"] == label]
    plot_dataframe(_df, label, cbar=False, output_pdf=f"Fig_7/tm_{label}.pdf", show=False)
