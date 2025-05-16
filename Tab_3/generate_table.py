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

import numpy as np
import numpy.ma as ma
import os

from lib.modulation.dvb_s2 import BPSK, QPSK, PSK_8, APSK_8_rate_100_180, APSK_16_rate_4_5, APSK_32_rate_4_5
from lib.modulation import QAM
from lib.misc import to_dB, from_dB, plot

os.makedirs(os.getcwd() + "/plot/Tab_3/", exist_ok=True)

constellations = {
    "BPSK": BPSK,
    "QPSK": QPSK,
    "8-PSK": PSK_8,
    "8-APSK": APSK_8_rate_100_180,
    "16-APSK": APSK_16_rate_4_5,
    "32-APSK": APSK_32_rate_4_5,
    "16-QAM": QAM(16),
    "64-QAM": QAM(64),
    "128-QAM": QAM(128),
    "256-QAM": QAM(256)
}

results = {}
for name, constellation in constellations.items():
    mins = []
    maxs = []
    avgs = []
    for b in range(1, constellation.bits_per_symbol()+1):
        distances = constellation.distances(n_differing_bits=b)
        mins.append(((ma.masked_invalid(distances).min()/2)**2)/b)
        maxs.append(((ma.masked_invalid(distances).max()/2)**2)/b)
        avgs += ((ma.masked_invalid(distances)/2)**2/b).compressed().tolist()

    # Find the min and max equivalent power per bit
    rho_min = to_dB(min(mins))
    rho_max = to_dB(max(maxs))
    rho_avg = to_dB(np.mean(avgs))

    # Sanity checks
    assert(rho_min <= rho_max)
    assert(rho_min <= rho_avg)
    assert(rho_avg <= rho_max)

    # Relate this to the overall power level of targeting a single codeword
    # in the min, max, and average cases
    results[name] = {
        "rho_min": rho_min,
        "rho_max": rho_max,
        "rho_avg": rho_avg,
        "mins": mins,
        "maxs": maxs,
    }

def format_latex_table(results):
    # Start the LaTeX tabular environment
    latex_table = r"\begin{tabular}{@{}cccccc@{}}" + "\n"
    latex_table += r"\toprule" + "\n"
    
    # Extract the keys (modulation types) for the header
    headers = " & ".join(results.keys())
    latex_table += " & " + headers + r" \\ \midrule" + "\n"
    
    # Prepare the rows for rho_min and rho_max
    rho_min_row = r"$\rho_\text{min}$ & " + " & ".join(f"{results[key]['rho_min']:.2f}" for key in results) + r" \\ " + "\n"
    rho_avg_row = r"$\rho_\text{avg}$ & " + " & ".join(f"{results[key]['rho_avg']:.2f}" for key in results) + r" \\ " + "\n"
    rho_max_row = r"$\rho_\text{max}$ & " + " & ".join(f"{results[key]['rho_max']:.2f}" for key in results) + r" \\ " + "\n"
    
    # Combine all parts
    latex_table += rho_min_row
    latex_table += rho_avg_row
    latex_table += rho_max_row
    latex_table += r"\bottomrule" + "\n"
    latex_table += r"\end{tabular}"
    
    return latex_table

with open("./plot/Tab_3/table.tex", 'w') as f:
    print(format_latex_table(results), file=f)
