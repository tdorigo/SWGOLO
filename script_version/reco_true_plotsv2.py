import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import copy
import warnings
import matplotlib.patches as patches
import argparse
import sys
from pathlib import Path
from matplotlib import colors
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from torch.utils.data import TensorDataset, DataLoader
from functions import shower_identification, Layouts, GenerateShowers, Reconstruction, DenormalizeLabels

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',type=str)
args = parser.parse_args()

set_name = Path(args.model_path).stem

model = Reconstruction()
model.load_state_dict(torch.load(args.model_path))
model.eval()

x, y = Layouts()
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

E_r = []
E_p = []

X_r = []
X_p = []

Y_r = []
Y_p = []

Th_r = []
Th_p = []

Ph_r = []
Ph_p = []

Type_r = []
Type_p = []

with torch.no_grad():
    for i in range(3000):
        Ne, Nm, Te, Tm, x0, y0, E, th, ph, shower_type = GenerateShowers(x, y)

        X = torch.tensor(
            np.column_stack((x, y, Ne, Nm, Te, Tm)),
            dtype=torch.float32
        ).view(1, -1)

        output = model(X)

        x_pred = output[0, 0].item()
        y_pred = output[0, 1].item()
        E_pred = output[0, 2]
        theta_pred = output[0, 3]
        sin_pred = output[0, 4]
        cos_pred = output[0, 5]
        value_pred = output[0, 6]

        E_pred, theta_pred, phi_pred = DenormalizeLabels(
            E_pred, theta_pred, sin_pred, cos_pred
        )
        type_pred = shower_identification(value_pred)

        E_p.append(float(E_pred))
        E_r.append(float(E.item()))

        X_r.append(float(x0.item()))
        X_p.append(float(x_pred * 5000.0))

        Y_r.append(float(y0.item()))
        Y_p.append(float(y_pred * 5000.0))

        Th_r.append(float(th.item()))
        Th_p.append(float(theta_pred))

        Ph_r.append(float(ph.item()))
        Ph_p.append(float(phi_pred))

        Type_r.append(shower_type)
        Type_p.append(type_pred)

        if (i + 1) % 1500 == 0:
            print(f"E Predicted: {float(E_pred):.1f}, E Real: {E.item():.1f}")

# convert to tensors
results = {
    "E_r": torch.tensor(E_r, dtype=torch.float32),
    "E_p": torch.tensor(E_p, dtype=torch.float32),
    "X_r": torch.tensor(X_r, dtype=torch.float32),
    "X_p": torch.tensor(X_p, dtype=torch.float32),
    "Y_r": torch.tensor(Y_r, dtype=torch.float32),
    "Y_p": torch.tensor(Y_p, dtype=torch.float32),
    "Th_r": torch.tensor(Th_r, dtype=torch.float32),
    "Th_p": torch.tensor(Th_p, dtype=torch.float32),
    "Ph_r": torch.tensor(Ph_r, dtype=torch.float32),
    "Ph_p": torch.tensor(Ph_p, dtype=torch.float32),
    "Type_r": Type_r,
    "Type_p": Type_p
}

torch.save(results, f"true_reco_{set_name}.pt")
print("true reco results saved")

#true_reco_results = torch.load(f"true_reco_{set_name}.pt")

#########
# PLOTS #
#########

Type_r = np.array(Type_r).ravel()
Type_p = np.array(Type_p).ravel()

# Energy plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
# --- MUONIC COMPONENT ---
idx_h = np.where(Type_r == "h")[0]
color_map = {"h": "green", "g": "red"}
colors = [color_map[t] for t in Type_p[idx_h]]

axs[0].scatter(np.array(E_r)[idx_h], np.array(E_p)[idx_h], alpha = .5, c = colors)
axs[0].plot(np.arange(.1, 10, .01), np.arange(.1, 10, .01), color = "black", linewidth = 4)
axs[0].set_title("Predictions of Proton Showers")
axs[0].set_xlabel("True E")
axs[0].set_ylabel("Predicted E")
axs[0].set_xlim(0.1, 10.1)
axs[0].set_ylim(0.1, 10.1)
axs[0].set_aspect('equal')

# --- ELECTROMAGNETIC COMPONENT ---
idx_g = np.where(Type_r == "g")[0]
color_map = {"g": "green", "h": "red"}
colors = [color_map[t] for t in Type_p[idx_g]]

axs[1].scatter(np.array(E_r)[idx_g], np.array(E_p)[idx_g], alpha = .5, c = colors)
axs[1].plot(np.arange(.1, 10, .01), np.arange(.1, 10, .01), color = "black", linewidth = 4)
axs[1].set_title("Predictions of Gamma Showers")
axs[1].set_xlabel("True E")
axs[1].set_ylabel("Predicted E")
axs[1].set_xlim(0.1, 10.1)
axs[1].set_ylim(0.1, 10.1)
axs[1].set_aspect('equal')

fig.suptitle("Energy Predictions of EASs")
plt.tight_layout()
plt.savefig(f"energy_{set_name}.png")
print('Energy true reco plot saved')

## --- E resolution ---
##E_true = true_reco_results["E_r"].numpy()
##E_reco = true_reco_results["E_p"].numpy()
#
#E_true=np.array(E_r)[idx_g]
#E_reco=np.array(E_p)[idx_g]
#mask = (E_reco > 0) & (E_true > 0)
#E_true = E_true[mask]
#E_reco = E_reco[mask]
#
#delta = E_reco - E_true
#
## Log-spaced bins
#bins = np.logspace(np.log10(E_true.min()), np.log10(E_true.max()), 30)
#
#bin_centers = []
#res_vals = []
#
#for i in range(len(bins) - 1):
#    m = (E_true >= bins[i]) & (E_true < bins[i + 1])
#
#    if np.sum(m) > 10:
#        E_bin = E_true[m]
#        delta_bin = delta[m]
#
#        # RMS in this bin
#        rms = np.sqrt(np.mean(delta_bin**2))
#
#        # Normalize by mean energy in bin (better than center)
#        mean_E = np.mean(E_bin)
#
#        bin_centers.append(np.sqrt(bins[i] * bins[i + 1]))
#        res_vals.append(rms / mean_E)
#
#bin_centers = np.array(bin_centers)
#res_vals = np.array(res_vals)
#
## Plot
#plt.figure()
#plt.scatter(bin_centers, res_vals)
#plt.xscale('log')
#plt.xlabel("E true")
#plt.ylabel("RMS(E_reco - E_true) / E_true")
##plt.title("Relative Energy Resolution")
#plt.grid(True)
#plt.savefig('energy_resolution_'+args.set_name+'.png')


# Shower core plot
fig, axs = plt.subplots(2, 2, figsize=(14, 14), sharex=True, sharey=True)

# --- MUONIC COMPONENT ---
idx_h = np.where(Type_r == "h")[0]
color_map = {"h": "green", "g": "red"}
colors = [color_map[t] for t in Type_p[idx_h]]

axs[0,0].scatter(np.array(X_r)[idx_h], np.array(X_p)[idx_h], alpha = .5, c = colors)
axs[0,0].plot(np.arange(-1000, 1000, 1), np.arange(-1000, 1000, 1), color = "black", linewidth = 4)
axs[0,0].set_title("X proton")
axs[0,0].set_xlabel("True X")
axs[0,0].set_ylabel("Predicted X")
axs[0,0].set_xlim(-1000, 1000)
axs[0,0].set_ylim(-1000, 1000)
axs[0,0].set_aspect('equal')

idx_h = np.where(Type_r == "h")[0]
color_map = {"h": "green", "g": "red"}
colors = [color_map[t] for t in Type_p[idx_h]]

axs[1,0].scatter(np.array(Y_r)[idx_h], np.array(Y_p)[idx_h], alpha = .5, c = colors)
axs[1,0].plot(np.arange(-1000, 1000, 1), np.arange(-1000, 1000, 1), color = "black", linewidth = 4)
axs[1,0].set_title("Y proton")
axs[1,0].set_xlabel("True Y")
axs[1,0].set_ylabel("Predicted Y")
axs[1,0].set_xlim(-1000, 1000)
axs[1,0].set_ylim(-1000, 1000)
axs[1,0].set_aspect('equal')

# --- ELECTROMAGNETIC COMPONENT ---
idx_g = np.where(Type_r == "g")[0]
color_map = {"g": "green", "h": "red"}
colors = [color_map[t] for t in Type_p[idx_g]]

axs[0,1].scatter(np.array(X_r)[idx_g], np.array(X_p)[idx_g], alpha = .5, c = colors)
axs[0,1].plot(np.arange(-1000, 1000, 1), np.arange(-1000, 1000, 1), color = "black", linewidth = 4)
axs[0,1].set_title("X gamma")
axs[0,1].set_xlabel("True X")
axs[0,1].set_ylabel("Predicted X")
axs[0,1].set_xlim(-1000, 1000)
axs[0,1].set_ylim(-1000, 1000)
axs[0,1].set_aspect('equal')

idx_g = np.where(Type_r == "g")[0]
color_map = {"g": "green", "h": "red"}
colors = [color_map[t] for t in Type_p[idx_g]]

axs[1,1].scatter(np.array(Y_r)[idx_g], np.array(Y_p)[idx_g], alpha = .5, c = colors)
axs[1,1].plot(np.arange(-1000, 1000, 1), np.arange(-1000, 1000, 1), color = "black", linewidth = 4)
axs[1,1].set_title("Y gamma")
axs[1,1].set_xlabel("True Y")
axs[1,1].set_ylabel("Predicted Y")
axs[1,1].set_xlim(-1000, 1000)
axs[1,1].set_ylim(-1000, 1000)
axs[1,1].set_aspect('equal')

fig.suptitle("Core cordinate predictions of EASs")
plt.tight_layout()
plt.savefig(f"showercore_{set_name}.png")
print('Shower core true reco plot saved')


# Theta plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# --- MUONIC COMPONENT ---
idx_h = np.where(Type_r == "h")[0]
color_map = {"h": "green", "g": "red"}
colors = [color_map[t] for t in Type_p[idx_h]]

axs[0].scatter(np.array(Th_r)[idx_h], np.array(Th_p)[idx_h], alpha = .5, c = colors)
axs[0].plot(np.arange(0, np.pi * 65 / 180, .01), np.arange(0, np.pi * 65 / 180, .01), color = "black", linewidth = 4)
axs[0].set_title("Predictions of Proton Showers")
axs[0].set_xlabel("True Th")
axs[0].set_ylabel("Predicted Th")
axs[0].set_xlim(0, np.pi * 65 / 180)
axs[0].set_ylim(0, np.pi * 65 / 180)
axs[0].set_aspect('equal')

# --- ELECTROMAGNETIC COMPONENT ---
idx_g = np.where(Type_r == "g")[0]
color_map = {"g": "green", "h": "red"}
colors = [color_map[t] for t in Type_p[idx_g]]

axs[1].scatter(np.array(Th_r)[idx_g], np.array(Th_p)[idx_g], alpha = .5, c = colors)
axs[1].plot(np.arange(0, np.pi * 65 / 180, .01), np.arange(0, np.pi * 65 / 180, .01), color = "black", linewidth = 4)
axs[1].set_title("Predictions of Gamma Showers")
axs[1].set_xlabel("True Th")
axs[1].set_ylabel("Predicted Th")
axs[1].set_xlim(0, np.pi * 65 / 180)
axs[1].set_ylim(0, np.pi * 65 / 180)
axs[1].set_aspect('equal')

fig.suptitle("Theta Predictions of EASs")
plt.tight_layout()
plt.savefig(f"theta_{set_name}.png")
print('theta true reco plot saved')


# azimuth plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# --- MUONIC COMPONENT ---
idx_h = np.where(Type_r == "h")[0]
color_map = {"h": "green", "g": "red"}
colors = [color_map[t] for t in Type_p[idx_h]]

axs[0].scatter(np.array(Ph_r)[idx_h], np.array(Ph_p)[idx_h], alpha = .5, c = colors)
axs[0].plot(np.arange(-np.pi, np.pi, .01), np.arange(-np.pi, np.pi, .01), color = "black", linewidth = 4)
axs[0].set_title("Predictions of Proton Showers")
axs[0].set_xlabel("True Ph")
axs[0].set_ylabel("Predicted Ph")
axs[0].set_xlim(-np.pi, np.pi)
axs[0].set_ylim(-np.pi, np.pi)
axs[0].set_aspect('equal')

# --- ELECTROMAGNETIC COMPONENT ---
idx_g = np.where(Type_r == "g")[0]
color_map = {"g": "green", "h": "red"}
colors = [color_map[t] for t in Type_p[idx_g]]

axs[1].scatter(np.array(Ph_r)[idx_g], np.array(Ph_p)[idx_g], alpha = .5, c = colors)
axs[1].plot(np.arange(-np.pi, np.pi, .01), np.arange(-np.pi, np.pi, .01), color = "black", linewidth = 4)
axs[1].set_title("Predictions of Gamma Showers")
axs[1].set_xlabel("True Ph")
axs[1].set_ylabel("Predicted Ph")
axs[1].set_xlim(-np.pi, np.pi)
axs[1].set_ylim(-np.pi, np.pi)
axs[1].set_aspect('equal')

fig.suptitle("Phi Predictions of EASs")
plt.tight_layout()
plt.savefig(f"phi_{set_name}.png")
print('phi true reco plot saved')

