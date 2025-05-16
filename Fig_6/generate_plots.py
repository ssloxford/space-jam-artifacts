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

# Generate Figure 6, real-world performance comparison

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
import json

from lib.modulation.dvb_s2 import BPSK, QPSK, PSK_8, APSK_8_rate_100_180, APSK_16_rate_4_5, APSK_32_rate_4_5
from lib.modulation import QAM
from lib.misc import from_dB, to_dB, plot
from lib.codeword_aware import generate_distance_distributions, optimal_error_rates

os.makedirs(os.getcwd() + "/plot/Fig_6/", exist_ok=True)

# Plotting helpers

def get_color(id, count):
    def lerp(a, b, x):
        return a * (1-x) + b * x
    x = (id+1) / count
    return ((lerp(1, 0, x)),(lerp(1, (33/255), x)),(lerp(1, (71/255), x)))

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Load the dataframes from the experiments

def load_and_concatenate_pickles(directory):
    """
    Load all pickle files in the specified directory and concatenate them into a single DataFrame.

    Parameters:
    directory (str): The path to the directory containing the pickle files.

    Returns:
    pd.DataFrame: A DataFrame containing the concatenated data from all pickle files.
    """
    dataframes = []
    
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.pkl') or filename.endswith('.pickle'):
            file_path = os.path.join(directory, filename)
            # Load the pickle file as a DataFrame
            df = pd.read_pickle(file_path)
            if "sync_plsc_diff.iq" in df.iloc[0].jammer_file and df.iloc[0].victim_file != "../video/tone_4_5.iq":
                print(filename)
            dataframes.append(df)
    
    # Concatenate all DataFrames in the list into a single DataFrame
    if dataframes:
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        return concatenated_df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no pickles were found

df_qpsk_4_5 = load_and_concatenate_pickles('./data/hardware/qpsk_4_5')
df_qpsk_9_10 = load_and_concatenate_pickles('./data/hardware/qpsk_9_10')

df_qpsk_4_5_noise = load_and_concatenate_pickles('./data/hardware/qpsk_4_5_noise')
df_qpsk_9_10_noise = load_and_concatenate_pickles('./data/hardware/qpsk_9_10_noise')
df = pd.concat((df_qpsk_4_5, df_qpsk_9_10))
df_noise = pd.concat((df_qpsk_4_5_noise, df_qpsk_9_10_noise))

def jammer_file_to_type(path):
    basename = path.split('/')[-1].rsplit('.', 1)[0]
    if "desync_pulsed" in basename:
        pulse_rate = float(basename.split("r=")[-1])
        return {
            "pulse_rate": pulse_rate,
            "jammer_type": "symbol",
            "targeted_segment": "frame",
            "phase": None
        }
    elif basename == "gaussian":
        return {
            "pulse_rate": 1,
            "jammer_type": "Gaussian",
            "targeted_segment": "frame",
            "phase": None
        }
    elif "sync_desync" in basename:
        return {
            "pulse_rate": 1,
            "jammer_type": "symbol",
            "targeted_segment": basename.split("_")[-1],
            "phase": None
        }
    elif "sync_gaussian" in basename:
        return {
            "pulse_rate": 1,
            "jammer_type": "Gaussian",
            "targeted_segment": basename.split("_")[-1],
            "phase": None
        }
    elif "sync_plsc_diff" in basename:
        return {
            "pulse_rate": 1,
            "jammer_type": "sync",
            "targeted_segment": "plsc",
            "phase": 0
        }
    elif basename == "sync_sof":
        return {
            "pulse_rate": 1,
            "jammer_type": "sync",
            "targeted_segment": "sof",
            "phase": 0
        }
    elif "sync_pulsed" in basename:
        match = re.search(r'sync_pulsed_r=([\d.]+)_a=([\d.]+)', basename)
        return {
            "pulse_rate": float(match.group(1)),
            "jammer_type": "sync",
            "targeted_segment": "data",
            "phase": float(match.group(2))
        }
    else:
        raise RuntimeError(f"jammer basename {basename} not recognized")
        
def victim_file_to_type(path):
    basename = path.split('/')[-1].split('.')[0]
    if basename == "tone_4_5":
        return {
            "encoded": "tone",
            "modulation": "QPSK",
            "rate": "4/5"
        }
    elif basename == "tone_9_10":
        return {
            "encoded": "tone",
            "modulation": "QPSK",
            "rate": "9/10"
        }
    elif basename == "tone_8psk_3_4":
        return {
            "encoded": "tone",
            "modulation": "8-PSK",
            "rate": "3/4"
        }
    else:
        print(basename)
        raise RuntimeError("victim basename not recognized")
    

def process_data(df):
    # Group by columns victim_file, jammer_file, Pj [dB]
    # For each group, take the average of the beep_correlation column
    result = df.groupby(['victim_file', 'jammer_file', 'Pj [dB]', 'sync'])['beep_correlation'].mean().reset_index()
    
    rows = []
    for index, row in result.iterrows():
        d = row.to_dict()
        try:
            d = {**d, **victim_file_to_type(d["victim_file"]), **jammer_file_to_type(d["jammer_file"])}
        except RuntimeError as e:
            continue
            
        rows.append(d)
    
    _df = pd.DataFrame(rows)
    
    # Filter rows where beep_correlation is below 600
    filtered_df = _df[_df['beep_correlation'] < 600]

    # Group by the specified columns and find the index of the row with the smallest Pj [dB]
    result = filtered_df.loc[filtered_df.groupby(["encoded", "modulation", "rate", "jammer_type", "targeted_segment", "pulse_rate"])['Pj [dB]'].idxmin()]
    
    # Add the class information
    result.loc[(result["jammer_type"] == "Gaussian") & (result["targeted_segment"] == "frame") & (result["pulse_rate"] == 1), "class"] = "Desynchronized"
    result.loc[(result["jammer_type"] == "symbol") & (result["targeted_segment"] == "frame"), "class"] = "Desynchronized"
    result.loc[(result["jammer_type"] == "symbol") & (result["targeted_segment"] != "frame"), "class"] = "Frame Synchronized"
    result.loc[(result["jammer_type"] == "Gaussian") & (result["targeted_segment"] != "frame") & (result["pulse_rate"] == 1), "class"] = "Gaussian Frame Synchronized"
    result.loc[(result["jammer_type"] == "sync") & (result["targeted_segment"] != "frame"), "class"] = "Synchronized"
    
    result["system"] = result.apply(lambda row: f"{row['modulation']} {row['rate']}", axis=1)    
    result["jammer"] = result.apply(lambda row: f"{row['targeted_segment']} {row['pulse_rate']}", axis=1)    
    
    return result

# Process the data

_df = process_data(df)
_df_noise = process_data(df_noise)

class_order = ["Desynchronized", "Frame Synchronized", "Synchronized",]
# Convert the 'class' column to a categorical type with the specified order
_df['class'] = pd.Categorical(_df['class'], categories=class_order, ordered=True)

system_order = ["QPSK 4/5", "QPSK 9/10"]
_df['system'] = pd.Categorical(_df['system'], categories=system_order, ordered=True)

# Sort the DataFrame by the 'class' column
_df_sorted = _df.sort_values(by='class').reset_index(drop=True)

# Convert the 'class' column to a categorical type with the specified order
_df_noise['class'] = pd.Categorical(_df_noise['class'], categories=class_order, ordered=True)

system_order = ["QPSK 4/5", "QPSK 9/10"]#, "8-PSK 3/4"]
_df_noise['system'] = pd.Categorical(_df_noise['system'], categories=system_order, ordered=True)

# Sort the DataFrame by the 'class' column
_df_sorted_noise = _df_noise.sort_values(by='class').reset_index(drop=True)

# Plotting
xlim = (0, -32)
fig_width = 5
fig_height = 10

num_classes = len(class_order)
num_systems = len(system_order)

def f(r):
    seg = r.targeted_segment
    rate = r.pulse_rate
    t = r.jammer_type
    if seg == "plsc":
        return "PLSC"
    elif seg == "header":
        return "Hdr; 1.00"
    elif seg == "sof":
        return "Preamble"
    elif seg == "data":
        return f"Data"
    elif t == "Gaussian":
        return f"Gaussian"
    else:
        return f"\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0Pulsed"

_df2 = _df_sorted.loc[_df_sorted.groupby(["jammer_type", "targeted_segment", "encoded", "modulation", "rate"])["Pj [dB]"].idxmin()].copy()
_df2_noise = _df_sorted_noise.loc[_df_sorted_noise.groupby(["jammer_type", "targeted_segment", "encoded", "modulation", "rate"])["Pj [dB]"].idxmin()].copy()

with open("./out/fig_6.json") as f2:
    table_values = json.load(f2)

# Iterate over each class
for i, cls in enumerate(class_order):
    @plot
    def plott():
        # Set the fig height based on the number of bars to be plotted
        if cls == "Desynchronized":
            _fig_height = fig_height*0.8/3.5
        elif cls == "Frame Synchronized":
            _fig_height = fig_height*0.8/3.5
        else:
            _fig_height = fig_height*1.15/3.5
        
        fig, axs = plt.subplots(num_systems, figsize=(fig_width, _fig_height/num_systems))

        d = _df2[(_df2["class"] == cls) ].reset_index()
        d_noise = _df2_noise[(_df2_noise["class"] == cls)].reset_index()

        for j, system in enumerate(system_order):
            ax = axs[j]
            pos = ax.get_position()
            color = get_color(j+1, num_systems + 2)

            d2 = d[d["system"] == system].reset_index()
            d2["tick"] = d2[["targeted_segment", "pulse_rate", "jammer_type"]].apply(f, axis=1)
            d2 = d2.sort_values(by='tick')
            if cls == "Desynchronized":
                ax.set_position([pos.x0, pos.y0, pos.width, pos.height])
                sns.barplot(data=d2, x='Pj [dB]', y='tick', orient='h', ax=ax, color=color)
            elif cls == "Gaussian":
                d2["tick2"] = "\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0"
                ax.set_position([pos.x0-0.056, pos.y0, pos.width, pos.height])
                sns.barplot(data=d2, x='Pj [dB]', y='tick2', orient='h', ax=ax, color=color)
            elif cls == "Frame Synchronized":
                for index, row in d2.iterrows():
                    if row.targeted_segment != "header":
                        ax.barh(row['tick'], row['Pj [dB]'], color=color)

            elif cls == "Synchronized":
                sns.barplot(data=d2, x='Pj [dB]', y='tick', orient='h', ax=ax, color=color)
            else:
                raise NotImplementedError()
            
            # Loop through each bar and draw a vertical line at the specified x-value
            try:
                stuff = table_values[cls][system]
                for i, bar in enumerate(ax.patches):
                    value = stuff[i]
                    y_top = bar.get_y()
                    y_bottom = bar.get_y() + bar.get_height()
                    ax.plot([value, value], [y_bottom, y_top], color='red', linestyle='solid')
            except KeyError:
                pass
    
            d2 = d_noise[d_noise["system"] == system].reset_index()
            d2["tick"] = d2[["targeted_segment", "pulse_rate", "jammer_type"]].apply(f, axis=1)
            d2 = d2.sort_values(by='tick')
            if cls == "Desynchronized":
                ax.set_position([pos.x0, pos.y0, pos.width, pos.height])
                for index, row in d2.iterrows():
                    ax.barh(row['tick'], row['Pj [dB]'], color=color, alpha=0.3, hatch='//', edgecolor='black')
            elif cls == "Gaussian":
                d2["tick2"] = "\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0"
                ax.set_position([pos.x0-0.056, pos.y0, pos.width, pos.height])
                sns.barplot(data=d2, x='Pj [dB]', y='tick2', orient='h', ax=ax, color=color, fill=False)
            elif cls == "Frame Synchronized":
                for index, row in d2.iterrows():
                    if row.targeted_segment != "header":
                        ax.barh(row['tick'], row['Pj [dB]'], color=color, alpha=0.3, hatch='//', edgecolor='black')
            elif cls == "Synchronized":
                for index, row in d2.iterrows():
                    ax.barh(row['tick'], row['Pj [dB]'], color=color, alpha=0.3, hatch='//', edgecolor='black')
            else:
                raise NotImplementedError()

        # Add legend
        if cls == "Desynchronized":
            colors = {
                "QPSK 4/5": get_color(1, num_systems + 2),
                "QPSK 9/10": get_color(2, num_systems + 2),
            }
            legend_handles = [mpatches.Patch(color=color, label=label) for label, color in colors.items()] + \
                [
                    mpatches.Patch(alpha=0.0, label=""),
                    mpatches.Patch(color="black", label=r"$N_0 = -\infty$dB"),
                    mpatches.Patch(facecolor="white", edgecolor="grey", hatch='//', label=r"$N_0 = -15$dB"),
                    Line2D([0], [0], color='red', linestyle='solid', label='Simulated Performance')
                ]
            plt.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.87, 4.5), ncol=2)

        monospace_font = FontProperties(family='monospace')
        # Styling
        for i, ax in enumerate(axs):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            if i != num_systems-1:
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim(xlim)

            yticks = ax.get_yticks()
            yticklabels = ax.get_yticklabels()

        fig.supylabel(cls if cls != "Frame Synchronized" else "Frame Sync", x=-0.07)
        fig.supxlabel("JSR [dB]", y=-0.18, fontsize=10)

    plott(output_pdf=f"Fig_6/{i}.pdf", show=False)
