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
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.font_manager
from pathlib import Path
import matplotlib
import os

def iq_base(xmin, xmax, ymin, ymax, below_ax_lambda = None):
    fig = plt.figure(figsize=((xmax - xmin) / (ymax - ymin) * 2, 2), frameon=False)
    ax = fig.add_subplot(1, 1, 1)

    plt.sca(ax)

    #Set limits
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    #Disable ticks
    plt.xticks([])
    plt.yticks([])

    ax.grid(False)
    ax.axis("equal")
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    # removing the default axis on all sides:
    for side in ["bottom", "right", "top", "left"]:
        ax.spines[side].set_visible(False)

    # manual arrowhead width and length
    hw = 1.2 / 8 * (ymax - ymin) / (5)
    hl = 1.5 / 8 * (ymax - ymin) / (5)
    lw = 1.0 # axis line width
    ohg = 0.4  # arrow overhang

    if below_ax_lambda is not None:
        below_ax_lambda()

    # compute matching arrowhead length and width
    
    # draw x and y axis
    ax.arrow(
        xmin + 0.3 * (ymax - ymin) / (5),
        0,
        xmax - xmin - 0.6 * (ymax - ymin) / (5),
        0.0,
        fc="k",
        ec="k",
        lw=lw,
        head_width=hw,
        head_length=hl,
        overhang=ohg,
        length_includes_head=True,
        clip_on=False,
    )

    ax.arrow(
        0,
        ymin + 0.3* (ymax - ymin) / (5),
        0.0,
        ymax - ymin - 0.6* (ymax - ymin) / (5),
        fc="k",
        ec="k",
        lw=lw,
        head_width=hw,
        head_length=hl,
        overhang=ohg,
        length_includes_head=True,
        clip_on=False,
    )
    plt.text(xmax - 0.3* (ymax - ymin) / (5), -0.5 * (ymax - ymin) / (5), "I")
    plt.text(0.08* (ymax - ymin) / (5), ymax - 0.4* (ymax - ymin) / (5), "Q")

# Ensure output directory exists
os.makedirs(os.getcwd() + "/plot/Fig_2/", exist_ok=True)

# Set style
sns.set_theme(context="notebook", style="white")
new_rc = {
    "xtick.bottom": False,
    "xtick.minor.bottom": False,
    "xtick.minor.visible": False,
    "xtick.labeltop": False,
    "xtick.labelbottom": False,
    "xtick.top": False,
    "ytick.left": False,
    "ytick.minor.left": False,
    "ytick.minor.visible": False,
    "ytick.right": False,
    "ytick.labelleft": False,
    "ytick.labelright": False,
    "grid.color": "#c0c0c0",
    "grid.linestyle": "-",
    "axes.grid": False,
    "font.family": "serif",
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
}
plt.rcParams |= new_rc

# Set colours

bad_color = [0.4, 0.4, 0.4]
good_color = [0.1, 0.9, 0.1]
cdict = {
    "red": [
        [0, bad_color[0], bad_color[0]],
        [0.5, bad_color[0], good_color[0]],
        [1, good_color[0], good_color[0]],
    ],
    "green": [
        [0, bad_color[1], bad_color[1]],
        [0.5, bad_color[1], good_color[1]],
        [1, good_color[1], good_color[1]],
    ],
    "blue": [
        [0, bad_color[2], bad_color[2]],
        [0.5, bad_color[2], good_color[2]],
        [1, good_color[2], good_color[2]],
    ],
    "alpha": [[0, 1, 1], [0.5, 0, 0], [1, 1, 1]],
}

newcmp = LinearSegmentedColormap("testCmap", segmentdata=cdict, N=256)

colcycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def draw_selection_regions():
    plt.plot([0, 0], [2, -2], c=(0.65,0.65,0.65))

# Define constants
EaEvdB = 3
EvN0dB = 6

EaN0dB = EaEvdB + EvN0dB

bpsk_constellation = np.array([1, -1])
qpsk_constellation = np.array([1+0j, -1+0j, 0+1j, 0-1j])

victim_constellation = bpsk_constellation

# Generate subfigure a)

iq_base(-2.5, 1.5, -1.5, 1.5, draw_selection_regions)
noise_x = np.arange(-3, 3, 0.02)
noise_y = np.arange(-3, 3, 0.02)
xx, yy = np.meshgrid(noise_x, noise_y)
Ea_and_N0dB = 10*np.log10(10 ** (-EvN0dB / 10) + 10**(EaEvdB / 10))
noise_sigma = 10 ** (Ea_and_N0dB / 20) / np.sqrt(2)

zzs = np.array(
    [
        np.exp(
            -((xx - np.real(victim_constellation[1])) ** 2 + (yy - np.imag(victim_constellation[0])) ** 2) / noise_sigma**2
        )
    ]
)
zz = np.sum(zzs, axis=0)
zzr = np.max(zz) * 1.2
zz *= np.where(xx < 0, 1, -1)
plt.contourf(xx, yy, zz, cmap=newcmp, vmin=-zzr, vmax=zzr, levels=16)

lines = []

lines.append(
    plt.scatter(
        np.real(victim_constellation),
        np.imag(victim_constellation),
        color=colcycle[0],
        s=25,
        marker="x"
    )
)

#plt.legend(lines, ["Attacker"])

plt.xlim(-2.5, 1.5)
plt.ylim(-1.5, 1.5)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("./plot/Fig_2/a.pdf")

# Generate subfigure b)

iq_base(-2.5, 1.5, -1.5, 1.5, draw_selection_regions)
noise_x = np.arange(-3, 3, 0.02)
noise_y = np.arange(-3, 3, 0.02)
xx, yy = np.meshgrid(noise_x, noise_y)
noise_sigma = 10 ** (-EvN0dB / 20) / np.sqrt(2)

zzs = np.array(
    [
        np.exp(
            -((xx - np.real(pt)) ** 2 + (yy - np.imag(pt)) ** 2) / noise_sigma**2
        )
        for pt in [
            victim_constellation[1]
            + bpsk_constellation[0] * 10 ** (EaEvdB / 20) * np.exp(1j * ph)
            for ph in np.linspace(0, 2 * np.pi, 600)
        ]
    ]
)
zz = np.sum(zzs, axis=0)
zzr = np.max(zz) * 1.2
zz *= np.where(xx < 0, 1, -1)
plt.contourf(xx, yy, zz, cmap=newcmp, vmin=-zzr, vmax=zzr, levels=16)

lines = []

lines.append(
    plt.scatter(
        np.real(victim_constellation),
        np.imag(victim_constellation),
        color=colcycle[0],
        s=25,
        marker="x"
    )
)

#plt.legend(lines, ["Attacker"])

plt.xlim(-2.5, 1.5)
plt.ylim(-1.5, 1.5)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("plot/Fig_2/b.pdf")

# Generate subfigure d)

iq_base(-2.5, 1.5, -1.5, 1.5, draw_selection_regions)
noise_x = np.arange(-3, 3, 0.02)
noise_y = np.arange(-3, 3, 0.02)
xx, yy = np.meshgrid(noise_x, noise_y)
noise_sigma = 10 ** (-EvN0dB / 20) / np.sqrt(2)

zzs = np.array(
    [
        np.exp(
            -((xx - np.real(pt)) ** 2 + (yy - np.imag(pt)) ** 2) / noise_sigma**2
        )
        for pt in [
            victim_constellation[1]
            + bpsk_constellation[1] * 10 ** (EaEvdB / 20) * np.exp(1j * np.pi),
            victim_constellation[1]
            - bpsk_constellation[1] * 10 ** (EaEvdB / 20) * np.exp(1j * np.pi)
        ]
    ]
)
zz = np.sum(zzs, axis=0)
zzr = np.max(zz) * 1.2
zz *= np.where(xx < 0, 1, -1)
plt.contourf(xx, yy, zz, cmap=newcmp, vmin=-zzr, vmax=zzr, levels=16)

lines = []

lines.append(
    plt.scatter(
        np.real(victim_constellation),
        np.imag(victim_constellation),
        color=colcycle[0],
        s=25,
        marker="x"
    )
)

#plt.legend(lines, ["Attacker"])

plt.xlim(-2.5, 1.5)
plt.ylim(-1.5, 1.5)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("plot/Fig_2/d.pdf")

# Generate subfigure e)

iq_base(-2.5, 1.5, -1.5, 1.5, draw_selection_regions)
noise_x = np.arange(-3, 3, 0.02)
noise_y = np.arange(-3, 3, 0.02)
xx, yy = np.meshgrid(noise_x, noise_y)
noise_sigma = 10 ** (-EvN0dB / 20) / np.sqrt(2)

zzs = np.array(
    [
        np.exp(
            -((xx - np.real(pt)) ** 2 + (yy - np.imag(pt)) ** 2) / noise_sigma**2
        )
        for pt in [
            victim_constellation[1]
            + bpsk_constellation[1] * 10 ** (EaEvdB / 20) * np.exp(1j * np.pi),
        ]
    ]
)
zz = np.sum(zzs, axis=0)
zzr = np.max(zz) * 1.2
zz *= np.where(xx < 0, 1, -1)
plt.contourf(xx, yy, zz, cmap=newcmp, vmin=-zzr, vmax=zzr, levels=16)

lines = []

lines.append(
    plt.scatter(
        np.real(victim_constellation),
        np.imag(victim_constellation),
        color=colcycle[0],
        s=25,
        marker="x"
    )
)

#plt.legend(lines, ["Attacker"])

plt.xlim(-2.5, 1.5)
plt.ylim(-1.5, 1.5)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("plot/Fig_2/e.pdf")
