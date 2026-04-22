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

#parser = argparse.ArgumentParser()
#parser.add_argument('--model_path',type=str)
#args = parser.parse_args()
#
#
#model = Reconstruction()
#model.load_state_dict(torch.load(args.model_path))
#model.eval()
#
#x, y = Layouts()
#x = torch.tensor(x, dtype=torch.float32)
#y = torch.tensor(y, dtype=torch.float32)
#
#E_r = []
#E_p = []
#
#X_r = []
#X_p = []
#
#Y_r = []
#Y_p = []
#
#Th_r = []
#Th_p = []
#
#Ph_r = []
#Ph_p = []
#
#Type_r = []
#Type_p = []
#
#with torch.no_grad():
#    for i in range(3000):
#        Ne, Nm, Te, Tm, x0, y0, E, th, ph, shower_type = GenerateShowers(x, y)
#
#        X = torch.tensor(
#            np.column_stack((x, y, Ne, Nm, Te, Tm)),
#            dtype=torch.float32
#        ).view(1, -1)
#
#        output = model(X)
#
#        x_pred = output[0, 0].item()
#        y_pred = output[0, 1].item()
#        E_pred = output[0, 2]
#        theta_pred = output[0, 3]
#        sin_pred = output[0, 4]
#        cos_pred = output[0, 5]
#        value_pred = output[0, 6]
#
#        E_pred, theta_pred, phi_pred = DenormalizeLabels(
#            E_pred, theta_pred, sin_pred, cos_pred
#        )
#        type_pred = shower_identification(value_pred)
#
#        E_p.append(float(E_pred))
#        E_r.append(float(E.item()))
#
#        X_r.append(float(x0.item()))
#        X_p.append(float(x_pred * 5000.0))
#
#        Y_r.append(float(y0.item()))
#        Y_p.append(float(y_pred * 5000.0))
#
#        Th_r.append(float(th.item()))
#        Th_p.append(float(theta_pred))
#
#        Ph_r.append(float(ph.item()))
#        Ph_p.append(float(phi_pred))
#
#        Type_r.append(shower_type)
#        Type_p.append(type_pred)
#
#        if (i + 1) % 1500 == 0:
#            print(f"E Predicted: {float(E_pred):.1f}, E Real: {E.item():.1f}")
#
## convert to tensors
#results = {
#    "E_r": torch.tensor(E_r, dtype=torch.float32),
#    "E_p": torch.tensor(E_p, dtype=torch.float32),
#    "X_r": torch.tensor(X_r, dtype=torch.float32),
#    "X_p": torch.tensor(X_p, dtype=torch.float32),
#    "Y_r": torch.tensor(Y_r, dtype=torch.float32),
#    "Y_p": torch.tensor(Y_p, dtype=torch.float32),
#    "Th_r": torch.tensor(Th_r, dtype=torch.float32),
#    "Th_p": torch.tensor(Th_p, dtype=torch.float32),
#    "Ph_r": torch.tensor(Ph_r, dtype=torch.float32),
#    "Ph_p": torch.tensor(Ph_p, dtype=torch.float32),
#    "Type_r": Type_r,
#    "Type_p": Type_p
#}
#
#torch.save(results, "./NN_Files/true_reco_results.pt")
#print("Saved reco results to ./NN_Files/reco_results.pt")

true_reco_results = torch.load("./NN_Files/true_reco_results.pt")

E_true = true_reco_results["E_r"].numpy()
E_reco = true_reco_results["E_p"].numpy()

mask = (E_reco > 0) & (E_true > 0)
E_true = E_true[mask]
E_reco = E_reco[mask]

# Absolute residual
delta = E_reco - E_true

# Log-spaced bins
bins = np.logspace(np.log10(E_true.min()), np.log10(E_true.max()), 30)

bin_centers = []
res_vals = []

for i in range(len(bins) - 1):
    m = (E_true >= bins[i]) & (E_true < bins[i + 1])

    if np.sum(m) > 10:
        E_bin = E_true[m]
        delta_bin = delta[m]

        # RMS in this bin
        rms = np.sqrt(np.mean(delta_bin**2))

        # Normalize by mean energy in bin (better than center)
        mean_E = np.mean(E_bin)

        bin_centers.append(np.sqrt(bins[i] * bins[i + 1]))
        res_vals.append(rms / mean_E)

bin_centers = np.array(bin_centers)
res_vals = np.array(res_vals)

# Plot
plt.figure()
plt.scatter(bin_centers, res_vals)
plt.xscale('log')
plt.xlabel("E true")
plt.ylabel("RMS(E_reco - E_true) / E_true")
plt.title("Relative Energy Resolution")
plt.grid(True)
plt.show()

exit()










# Compute dispersion
disp = 1 - (E_true / E_reco)

bins = np.logspace(np.log10(E_true.min()), np.log10(E_true.max()), 30)

bin_centers = []
disp_mean = []
disp_std = []

for i in range(len(bins) - 1):
    mask = (E_true >= bins[i]) & (E_true < bins[i+1])
    
    if np.sum(mask) > 10:  # avoid low statistics
        bin_centers.append(np.sqrt(bins[i]*bins[i+1]))  # geometric center
        disp_mean.append(np.mean(disp[mask]))
        disp_std.append(np.std(disp[mask]))

bin_centers = np.array(bin_centers)
disp_mean = np.array(disp_mean)
disp_std = np.array(disp_std)

# Plot
plt.figure()
plt.errorbar(bin_centers, disp_mean, yerr=disp_std, fmt='o', capsize=3)
plt.xscale('log')
plt.xlabel("E true")
plt.ylabel("1 - (E_true / E_reco)")
plt.title("Energy dispersion vs E true")
plt.grid()
plt.show()

exit()














mask = (E_reco > 1)
E_true = E_true[mask]
E_reco = E_reco[mask]

# Compute dispersion
disp = 1 - (E_true / E_reco)

# --- Scatter plot (optional, can be dense) ---
plt.figure()
plt.scatter(E_true, disp, s=5, alpha=0.3)
plt.xlabel("E true")
plt.ylabel("1 - (E_true / E_reco)")
plt.title("Energy dispersion (scatter)")
plt.grid()
plt.show()

exit()


mask = E_p > 0
E_true = E_r[mask]
E_reco = E_p[mask]

disp = 1.0 - (E_true / E_reco)

bins = np.logspace(np.log10(E_true.min()), np.log10(E_true.max()), 20)
bin_centers = []
rms_vals = []

for i in range(len(bins) - 1):
    m = (E_true >= bins[i]) & (E_true < bins[i + 1])
    if np.sum(m) > 10:
        bin_centers.append(np.sqrt(bins[i] * bins[i + 1]))
        rms_vals.append(np.sqrt(np.mean(disp[m]**2)))

bin_centers = np.array(bin_centers)
rms_vals = np.array(rms_vals)

plt.figure()
plt.plot(bin_centers, rms_vals, "o-")
plt.xscale("log")
plt.xlabel("True Energy")
plt.ylabel(r"RMS of $1 - E_{\mathrm{true}}/E_{\mathrm{reco}}$")
plt.grid(True)
plt.show()
