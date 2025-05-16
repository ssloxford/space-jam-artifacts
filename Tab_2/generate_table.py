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
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm
import json

os.makedirs(os.getcwd() + "/plot/Tab_2/", exist_ok=True)

from pprint import pprint

def optimal_error_rate(Pj, N0, threshold):
    std_dev = np.sqrt(N0 / 2)
    ts = threshold - np.sqrt(Pj)
    cdf_values = norm.cdf(ts, loc=0, scale=std_dev)
    error_rate = 1 - cdf_values
    return error_rate

# Values from Table 1

TC_efficiency_dB = -5.44
TM_efficiency_dB = -0.05
DVB_S2_efficiency_dB = -0.05
DVB_S2_efficiency_dB_plscode = -27.11

# Calculate full knowledge table section

# These are the distances to the boundary and size of the codeword
# Final performance metrics given by also including the other factors
# of framing gain and deinterleaving gain
distances = {
    "TC": (np.sqrt(14), 128), # Min dist from p31 of https://ccsds.org/Pubs/230x1g3e1.pdf
    "TM": (np.sqrt(27*0.5), 5120), # Min dist from p10 of https://arxiv.org/pdf/1201.2386, *0.5 since QPSK
    "DVB-S2": (np.sqrt(32), 64), # Min dist from DVB-S2 standard
}

half_dB = 10*np.log10(49/99)

N0 = from_dB(-15)
Pjs = from_dB(np.arange(-20, 20, 0.1)) # Can increase precision to 0.01

fk_dict = {}

for name, (threshold, n) in distances.items():
    errors = optimal_error_rate(Pjs, N0, threshold)
    idx_100 = np.argmax(errors > 0.99)
    Pj_100 = to_dB(Pjs[idx_100] / n)
    fk_dict[name] = Pj_100

tc_fk_dB = fk_dict["TC"]
tm_fk_dB = fk_dict["TM"]
dvb_s2_fk_dB = fk_dict["DVB-S2"]

# Next

def max_error_rate(_df, label):
    __df = _df[_df["label"] == label]
    idx = __df.groupby(['label', 'JSR'])['error_rate'].idxmax()
    result_df = __df.loc[idx].reset_index(drop=True)
    return result_df

def find_lowest_above_x(df, x, column):
    # Filter the DataFrame for values greater than x
    filtered_df = df[df[column] >= x]
    
    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        return None
    
    # Find the row with the minimum value in the filtered DataFrame
    min_row = filtered_df.loc[filtered_df[column].idxmin()]
    
    # Check if this minimum value is the overall minimum in the original DataFrame
    if min_row[column] == df[column].min():
        return None
    
    return min_row

def get_table_results(df, gain_dB, full_knowledge_power_dB, N, plsc_dB=None):
    """
    Returns the results to be inserted into the LaTeX table, for a jammer described by a given dataframe
    """
    
    _df = df.copy() # Ensure we don't overwrite df by a side-effect by mistake
    
    dps=2 # number of decimals to round to
    rnd = lambda x: np.round(x, dps)
    half_dB = 10*np.log10(49/99)
    
    ## Calculate relevant error rates
    _df["error_rate_interleaved"] = 1 - (1 - _df["error_rate"]) ** N
    _df["JSR_adjusted_dB"] = _df["JSR_dB"] + to_dB(1/N)
    _df["JSR_adjusted_2_dB"] = _df["JSR_dB"] + gain_dB + to_dB(1/N)
    
    ## Gaussian
    gaussian = _df[(_df["label"] == "desync_gaussian") & (_df["pulse_rate"] == 1)]
    gaussian_99 = find_lowest_above_x(gaussian, 0.99, column='error_rate_interleaved')
    gaussian_50 = find_lowest_above_x(gaussian, 0.49, column='error_rate_interleaved')

    ## Interleaved
    interleaved = max_error_rate(_df, "desync_symbol")
    interleaved_99 = find_lowest_above_x(interleaved, 0.99, column='error_rate_interleaved')
    interleaved_50 = find_lowest_above_x(interleaved, 0.49, column='error_rate_interleaved')

    ## Deinterleaved
    deinterleaved = max_error_rate(_df, "desync_symbol")
    deinterleaved_99 = find_lowest_above_x(deinterleaved, 0.99, column='error_rate')

    ## Coherent
    coherent = max_error_rate(_df, "sync_symbol")
    coherent_99 = find_lowest_above_x(coherent, 0.99, column='error_rate')
    
    return {
        "0_gaussian": {
            "100%": rnd(gaussian_99["JSR_dB"]) if gaussian_99 is not None else None,
            "50%": rnd(gaussian_50["JSR_dB"]) if gaussian_50 is not None else None,
        },
        "1_interleaved": {
            "100%": rnd(interleaved_99["JSR_dB"]) if interleaved_99 is not None else None,
            "50%": rnd(interleaved_50["JSR_dB"]) if interleaved_50 is not None else None,
        },
        "2_deinterleaved": {
            "100%": rnd(deinterleaved_99["JSR_adjusted_dB"]) if deinterleaved_99 is not None else None,
            "50%": rnd(deinterleaved_99["JSR_adjusted_dB"] + half_dB) if deinterleaved_99 is not None else None,
        },
        "3_frame_sync": {
            "100%": rnd(deinterleaved_99["JSR_adjusted_2_dB"]) if deinterleaved_99 is not None else None,
            "50%": rnd(deinterleaved_99["JSR_adjusted_2_dB"] + half_dB) if deinterleaved_99 is not None else None,
        },
        "4_coherent": {
            "100%": rnd(coherent_99["JSR_adjusted_2_dB"]) if coherent_99 is not None else None,
            "50%": rnd(coherent_99["JSR_adjusted_2_dB"] + half_dB) if coherent_99 is not None else None,
        },
        "5_full_knowledge": {
            "100%": full_knowledge_power_dB + (gain_dB if plsc_dB is None else plsc_dB) + to_dB(1/N),
            "50%": full_knowledge_power_dB + (gain_dB if plsc_dB is None else plsc_dB) + to_dB(1/N) + half_dB,
        }
    }

def format_data_rows(data, target="", threshold=-3):
    # Initialize the rows
    rows = []

    # Function to calculate differences
    def calculate_row_with_differences(row_values):
        differences = []
        previous_value = None
        for value in row_values:
            if value is None or previous_value is None:
                difference = '-'
            else:
                difference = value - previous_value
            
            # Format the current value with a prefixed + if non-negative
            formatted_value = f"{value:.2f}" if value is not None else '-'
            if difference != '-':
                # Check if the difference is less than the threshold
                if difference < threshold:
                    # Make the bracketed value bold
                    differences.append(f"{formatted_value} \\textit{{\\textbf{{({difference:+.2f})}}}}")
                else:
                    differences.append(f"{formatted_value} \\textit{{({difference:+.2f})}}")
            else:
                differences.append(f"{formatted_value} \\textit{{({difference})}}")
            if value is not None:
                previous_value = value
        return differences

    # Extract the 100% row
    row_100 = [r"$\sim$100\%"]
    row_100_values = []
    for key in sorted(data.keys()):
        value = data[key]['100%']
        row_100_values.append(value)  # Use 0 for None
    row_100_differences = calculate_row_with_differences(row_100_values)
    rows.append("& " + " & ".join([target, row_100[0]] + row_100_differences) + r" \\")

    # Extract the 50% row
    row_50 = [r"$\sim$50\%"]
    row_50_values = []
    for key in sorted(data.keys()):
        value = data[key]['50%']
        row_50_values.append(value)  # Use 0 for None
    row_50_differences = calculate_row_with_differences(row_50_values)
    rows.append("& " + " & ".join([target, row_50[0]] + row_50_differences) + r"\\")

    return "\n".join(rows)

# Load the data
df_tc = pd.read_pickle("./data/df_i.pkl")
#df_tm = pd.read_pickle("df_tm_i.pkl")
df_tm = pd.read_pickle("./data/tm_4_5_short.pkl")
df_dvbs2_4_5 = pd.read_pickle("./data/dvbs2_4_5_short.pkl")
df_dvbs2_9_10 = pd.read_pickle("./data/dvbs2_9_10_short.pkl")

with open("./out/preamble_TM.json") as f:
    preamble_TM = json.load(f)

with open("./out/preamble_TC.json") as f:
    preamble_TC = json.load(f)

with open("./out/preamble_DVB-S2.json") as f:
    preamble_DVB_S2 = json.load(f)

# Print the table
with open("./plot/Tab_2/table.tex", 'w') as f:
    print(
r"""
\begin{tabular}{@{}ccccccccc@{}}
\toprule
&&& \multicolumn{3}{c}{\textbf{Desynchronized}} & \multicolumn{1}{c}{} & \multicolumn{1}{c}{} & \multicolumn{1}{c}{} \\ 
\cmidrule(lr){4-6}
\textbf{Protocol} & \textbf{Target} & \textbf{FER} & \textbf{Gaussian} & \textbf{Pulsed Symb.} & \textbf{Deinterleaved} & \textbf{Frame sync} & \textbf{Synchronized} & \textbf{Full knowledge} \\
\midrule
\multirow{4}{*}{\makecell{\textbf{CCSDS TC} $\mathbf{(128, 68)}$ \\ $CFR = 16$; BPSK \\ $d_\text{min} = 14$~\cite{ccsds_230.1_G_3}; $d_\text{pre} = \SI{64}{\bit}$}}
""", file=f
    )

    print(format_data_rows(get_table_results(df_tc, TC_efficiency_dB, tc_fk_dB, 16), target="Code"), file=f)
    print(format_data_rows(preamble_TC, target="Preamble"), file=f)

    print(
r"""
\midrule
\multirow{4}{*}{\makecell{\textbf{CCSDS TM} $\mathbf{r=4/5}$ \\ $CFR = 16$; QPSK \\ $d_\text{min} = 27$ $\dagger$~\cite{butler2013bounds}; $d_\text{pre} = \SI{64}{\bit}$}}
""", file=f
    )

    print(format_data_rows(get_table_results(df_tm, TM_efficiency_dB, tm_fk_dB, 16), target="Code"), file=f)
    print(format_data_rows(preamble_TM, target="Preamble"), file=f)

    print(
r"""
\midrule
\multirow{4}{*}{\makecell{\textbf{DVB-S2} $\mathbf{r=4/5}$ \\ $CFR = 1$; QPSK \\ $d_\text{min} = 32$; $d_\text{pre} = \SI{26}{\symbol}$}}
""", file=f
    )

    print(format_data_rows(get_table_results(df_dvbs2_4_5, DVB_S2_efficiency_dB, dvb_s2_fk_dB, 1, DVB_S2_efficiency_dB_plscode), target="Code"), file=f)
    print(format_data_rows(preamble_DVB_S2, target="Preamble"), file=f)

    print(
r"""
\midrule
\multirow{4}{*}{\makecell{\textbf{DVB-S2} $\mathbf{r=9/10}$ \\ $CFR = 1$; QPSK \\ $d_\text{min} = 32$; $d_\text{pre} = \SI{26}{\symbol}$}}
""", file=f
    )

    print(format_data_rows(get_table_results(df_dvbs2_9_10, DVB_S2_efficiency_dB, dvb_s2_fk_dB, 1, DVB_S2_efficiency_dB_plscode), target="Code"), file=f)
    print(format_data_rows(preamble_DVB_S2, target="Preamble"), file=f)

    print(
r"""
\bottomrule
\end{tabular}
""", file=f
    )

# Export values for Figure 6

qpsk_4_5 = get_table_results(df_dvbs2_4_5, DVB_S2_efficiency_dB, dvb_s2_fk_dB, 1, DVB_S2_efficiency_dB_plscode)
qpsk_9_10 = get_table_results(df_dvbs2_9_10, DVB_S2_efficiency_dB, dvb_s2_fk_dB, 1, DVB_S2_efficiency_dB_plscode)

## Fill this according to the format Figure 6 requires
table_values = {
    "Desynchronized": {
        "QPSK 4/5":
            [qpsk_4_5['0_gaussian']['100%'], qpsk_4_5['1_interleaved']['100%']],
        "QPSK 9/10": 
            [qpsk_9_10['0_gaussian']['100%'], qpsk_9_10['1_interleaved']['100%']],
    },
    "Frame Synchronized": {
        "QPSK 4/5":
            [None, None],
        "QPSK 9/10": 
            [None, None],
    },
    "Synchronized": {
        "QPSK 4/5":
            [qpsk_4_5['4_coherent']['100%'], qpsk_4_5['5_full_knowledge']['100%'], preamble_DVB_S2['5_full_knowledge']['100%']],
        "QPSK 9/10":
            [qpsk_9_10['4_coherent']['100%'], qpsk_9_10['5_full_knowledge']['100%'], preamble_DVB_S2['5_full_knowledge']['100%']],
    },
}

with open(f"./out/fig_6.json", "w") as f:
    json.dump(table_values, f, indent=4)
