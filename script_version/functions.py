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
import sys
from pathlib import Path
from matplotlib import colors
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from torch.utils.data import TensorDataset, DataLoader



# GLOBAL VARIABLES USED IN FUNCTIONS
#Matrices we use in the functions
path_g = "/big_disk_2/swgo_test/SWGOLO/Fit_Photon_10_pars.txt"
path_p = "/big_disk_2/swgo_test/SWGOLO/Fit_Proton_2_pars.txt"
A = torch.tensor([[1, 1, 1, 1],[1, 2, 4, 8], [1, 3, 9, 27], [1, 4, 16, 64]] , dtype = torch.float32)

#Constants
c0 = .29979 #Speed of light in [m / ns] units
theta_max = np.pi * 15 / 180
log_01 = torch.tensor([np.log(.1)], dtype = torch.float32)
log_10 = torch.tensor([np.log(10)], dtype = torch.float32)
sqrt12 = torch.tensor([np.sqrt(12)], dtype = torch.float32)

#Tank Values
IntegrationWindow = 128. #128 ns integration window, SWGO default
sigma_time = 10. #Time resolution assumed for the detectors
R_min = 2.
TankArea = 68.59 * np.pi #Area for 19 hexagonal macro unit
TankRadius = np.sqrt(68.59) #Radius of macro unit

#Background
Bgr_mu_per_m2 = 0.000001826 * IntegrationWindow
fluxB_m = torch.tensor([TankArea * Bgr_mu_per_m2])

Bgr_e_per_m2 = 0.000000200 * IntegrationWindow
fluxB_e = torch.tensor([TankArea * Bgr_e_per_m2])

#Sizes
Nunits = 90
RelResCounts = .05

#Debug Parameters
largenumber = 1e13
epsilon = 1 / largenumber

#NN
Nevents = 400000 
Nvalidation = 40000

# LO-Optimization 
SWGOopt = False



# SHOWER GENERATOR 
'''
These functions read the parametrized g/p files to generate the inputs (core, flux, arrival times, energy, direction, type_particle) 
to train the NN.
'''
def ReadShowers(path_g, path_p):
    #GAMMA SHOWERS
    #Reading the Electron Parameters in the Showers
    PXeg1_p = np.loadtxt(path_g, max_rows = 3)

    for i in range(3):
        if PXeg1_p[i, 0]*PXeg1_p[i, 1]*PXeg1_p[i, 2] == 0:
            warnings.warn("Encountered 0")
            return

    PXeg2_p = np.loadtxt(path_g, skiprows = 3, max_rows = 3)

    for i in range(3):
        if PXeg2_p[i, 0]*PXeg2_p[i, 1]*PXeg2_p[i, 2] == 0:
            warnings.warn("Encountered 0")
            return

    PXeg3_p = np.loadtxt(path_g, skiprows = 6, max_rows = 3)

    for i in range(3):
        if PXeg3_p[i, 0]*PXeg3_p[i, 1]*PXeg3_p[i, 2] == 0:
            warnings.warn("Encountered 0")
            return

    PXeg4_p = np.loadtxt(path_g, skiprows = 9, max_rows = 3)

    for i in range(3):
        if PXeg4_p[i, 0]*PXeg4_p[i, 1]*PXeg4_p[i, 2] == 0:
            warnings.warn("Encountered 0")
            return

    #Reading the Muon Parameters in the Showers
    PXmg1_p = np.loadtxt(path_g, skiprows = 12, max_rows = 3)

    for i in range(3):
        if PXmg1_p[i, 0]*PXmg1_p[i, 1]*PXmg1_p[i, 2] == 0 and i != 1:
            warnings.warn("Encountered 0")
            return

    PXmg2_p = np.loadtxt(path_g, skiprows = 15, max_rows = 3)

    for i in range(3):
        if PXmg2_p[i, 0]*PXmg2_p[i, 1]*PXmg2_p[i, 2] == 0 and i != 1:
            warnings.warn("Encountered 0")
            return

    PXmg3_p = np.loadtxt(path_g, skiprows = 18, max_rows = 3)

    for i in range(3):
        if PXmg3_p[i, 0]*PXmg3_p[i, 1]*PXmg3_p[i, 2] == 0 and i != 1:
            warnings.warn("Encountered 0")
            return

    PXmg4_p = np.loadtxt(path_g, skiprows = 21, max_rows = 3)

    for i in range(3):
        if PXmg4_p[i, 0]*PXmg4_p[i, 1]*PXmg4_p[i, 2] == 0 and i != 1:
            warnings.warn("Encountered 0")
            return

    #PROTON SHOWERS
    #Reading the Electron Parameters in the Showers
    PXep1_p = np.loadtxt(path_p, max_rows = 3)

    for i in range(3):
        if PXep1_p[i, 0]*PXep1_p[i, 1]*PXep1_p[i, 2] == 0:
            warnings.warn("Encountered 0")
            return

    PXep2_p = np.loadtxt(path_p, skiprows = 3, max_rows = 3)

    for i in range(3):
        if PXep2_p[i, 0]*PXep2_p[i, 1]*PXep2_p[i, 2] == 0:
            warnings.warn("Encountered 0")
            return

    PXep3_p = np.loadtxt(path_p, skiprows = 6, max_rows = 3)

    for i in range(3):
        if PXep3_p[i, 0]*PXep3_p[i, 1]*PXep3_p[i, 2] == 0:
            warnings.warn("Encountered 0")
            return

    PXep4_p = np.loadtxt(path_p, skiprows = 9, max_rows = 3)

    for i in range(3):
        if PXep4_p[i, 0]*PXep4_p[i, 1]*PXep4_p[i, 2] == 0:
            warnings.warn("Encountered 0")
            return

    #Reading the Muon Parameters in the Showers
    PXmp1_p = np.loadtxt(path_p, skiprows = 12, max_rows = 3)

    for i in range(3):
        if PXmp1_p[i, 0]*PXmp1_p[i, 1]*PXmp1_p[i, 2] == 0 and i != 1:
            warnings.warn("Encountered 0")
            return

    PXmp2_p = np.loadtxt(path_p, skiprows = 15, max_rows = 3)

    for i in range(3):
        if PXmp2_p[i, 0]*PXmp2_p[i, 1]*PXmp2_p[i, 2] == 0 and i != 1:
            warnings.warn("Encountered 0")
            return

    PXmp3_p = np.loadtxt(path_p, skiprows = 18, max_rows = 3)

    for i in range(3):
        if PXmp3_p[i, 0]*PXmp3_p[i, 1]*PXmp3_p[i, 2] == 0 and i != 1:
            warnings.warn("Encountered 0")
            return

    PXmp4_p = np.loadtxt(path_p, skiprows = 21, max_rows = 3)

    for i in range(3):
        if PXmp4_p[i, 0]*PXmp4_p[i, 1]*PXmp4_p[i, 2] == 0 and i != 1:
            warnings.warn("Encountered 0")
            return

    PXmg_p = torch.tensor([PXmg1_p, PXmg2_p, PXmg3_p, PXmg4_p])
    PXeg_p = torch.tensor([PXeg1_p, PXeg2_p, PXeg3_p, PXeg4_p])
    PXmp_p = torch.tensor([PXmp1_p, PXmp2_p, PXmp3_p, PXmp4_p])
    PXep_p = torch.tensor([PXep1_p, PXep2_p, PXep3_p, PXep4_p])

    return PXmg_p, PXeg_p, PXmp_p, PXep_p


# We define PXmg_p, PXeg_p, PXmp_p, PXep_p since is used in the following functions
PXmg_p, PXeg_p, PXmp_p, PXep_p = ReadShowers(path_g, path_p)


def Y(energy, f_mode):
    Y = torch.zeros((4, 1))

    #Convert Energy to PeV
    xe = .5 + 20 * (torch.log(energy) - log_01) / (log_10 - log_01)
    xe2 = xe * xe

    #Evaluate Y for given energy and mode
    for i in range(4):
        if f_mode == "mg0":
            Y[i] = torch.exp(PXmg_p[i, 0, 0]) + torch.exp(PXmg_p[i ,0, 1] * torch.pow(xe, PXmg_p[i, 0, 2]))

        elif f_mode == "mg2":
            Y[i] = PXmg_p[i, 2, 0] + PXmg_p[i, 2, 1] * xe + PXmg_p[i, 2, 2] * xe2

        elif f_mode == "eg0":
            Y[i] = PXeg_p[i, 0, 0] * torch.exp(PXeg_p[i, 0, 1] * torch.pow(xe, PXeg_p[i, 0, 2]))

        elif f_mode == "eg1":
            Y[i] = PXeg_p[i, 1, 0] + PXeg_p[i, 1, 1] * xe + PXeg_p[i, 1, 2] * xe2

        elif f_mode == "eg2":
            Y[i] = PXeg_p[i, 2, 0] + PXeg_p[i, 2, 1] * xe + PXeg_p[i, 2, 2] * xe2

        elif f_mode == "mp0":
            Y[i] = torch.exp(PXmp_p[i, 0, 0]) + torch.exp(PXmp_p[i, 0, 1] * torch.pow(xe, PXmp_p[i, 0, 2]))

        elif f_mode == "mp2":
            Y[i] = PXmp_p[i, 2, 0] + PXmp_p[i, 2, 1] * xe + PXmp_p[i, 2, 2] * xe2

        elif f_mode == "ep0":
            Y[i] = torch.exp(PXep_p[i, 0, 0]) + torch.exp(PXep_p[i, 0, 1] * torch.pow(xe, PXep_p[i, 0, 2]))

        elif f_mode == "ep1":
            Y[i] = PXep_p[i, 1, 0] + PXep_p[i, 1, 1] * xe + PXep_p[i, 1, 2] * xe2

        elif f_mode == "ep2":
            Y[i] = PXep_p[i, 2, 0] + PXep_p[i, 2, 1] * xe + PXep_p[i, 2, 2] * xe2

        else:
            warnings.warn("The Mode is not defined")
            return

    return Y


def thisp(energy, theta, f_mode):
    #Find Y values
    Y_val = Y(energy, f_mode)

    #Solve for B
    B = torch.linalg.solve(A, Y_val)

    #Define x from theta
    x = .5 + 4 * theta / theta_max

    res = 0

    for i in range(4):
        res += B[i] * x ** i

    return res


def ShowerContent(energy, theta, R, f_mode):
    #Check if everything is in the range:
    R = torch.clamp(R, min=R_min)

    if energy < .1 or energy > 10:
        warnings.warn("Energy is out of range!")
        return 0

    if theta < 0 or theta > theta_max:
        warnings.warn("Angle is out of range!")
        return 0

    if f_mode == "mg" or f_mode == "mp":
        thisp0 = thisp(energy, theta, f_mode + "0")
        thisp2 = thisp(energy, theta, f_mode + "2")

        #Evaluate the  Flux
        flux0 = TankArea * .02 * thisp0 * torch.exp(-1 * torch.pow(R, thisp2))

    elif f_mode == "eg" or f_mode == "ep":
        thisp0 = thisp(energy, theta, f_mode + "0")
        thisp1 = thisp(energy, theta, f_mode + "1")
        thisp2 = thisp(energy, theta, f_mode + "2")

        #Evaluate the Flux
        flux0 = TankArea * thisp0 * torch.exp(-thisp1 * torch.pow(R, thisp2))

    else:
        warnings.warn("Mode is not defined")
        return

    #We need to make sure that flux is non negative or too large
    flux0 = torch.clamp(flux0, min=epsilon, max=largenumber)

    return flux0

def Layouts():
    R = [50, 200, 350, 550, 750, 1000]
    x = []
    y = []

    for i, r in enumerate(R):
        if i == 0:
            for j in range(3):
                x.append(r * np.cos(j *np.pi * 2 / 3))
                y.append(r * np.sin(j *np.pi * 2 / 3))

        if i == 1:
            for j in range(9):
                x.append(r * np.cos(j * np.pi * 2 / 9))
                y.append(r * np.sin(j * np.pi * 2 / 9))

        if i == 2:
            for j in range(12):
                x.append(r * np.cos(j * np.pi * 2 / 12))
                y.append(r * np.sin(j * np.pi * 2 / 12))

        if i == 3:
            for j in range(15):
                x.append(r * np.cos(j * np.pi * 2 / 15))
                y.append(r * np.sin(j * np.pi * 2 / 15))

        if i == 4:
            for j in range(18):
                x.append(r * np.cos(j * np.pi * 2 / 18))
                y.append(r * np.sin(j * np.pi * 2 / 18))

        if i == 5:
            for j in range(33):
                x.append(r * np.cos(j * np.pi * 2 / 33))
                y.append(r * np.sin(j * np.pi * 2 / 33))

    return x, y

def GenerateShowers(x, y):
    #Randomize Hadronic / Gamma Showers
    shower_r = torch.tensor([np.random.rand()], dtype = torch.float32)

    if shower_r >= .5:
        shower_type = "h"
    else:
        shower_type = "g"

    #Define the position of the shower cores
    X0 = torch.tensor([np.random.uniform(-1000, 1000)], dtype = torch.float32)
    Y0 = torch.tensor([np.random.uniform(-1000, 1000)], dtype = torch.float32)

    #Define the energy and angles
    energy = torch.tensor([np.random.uniform(.1, 10)], dtype = torch.float32)
    theta = torch.tensor([np.random.uniform(0, theta_max)], dtype = torch.float32)
    phi = torch.tensor([np.random.uniform(-np.pi, np.pi)], dtype = torch.float32)

    #Debugging part
    if SWGOopt == True:
        #Real distribution of Hadronic and Gamma Showers:
        shower = hg_dist()
        #Shower Core is at origin
        X0 = torch.tensor([np.random.uniform(-3000, 3000)], dtype = torch.float32)
        Y0 = torch.tensor([np.random.uniform(-3000, 3000)], dtype = torch.float32)
        #Energy is 1 PeV
        energy = torch.tensor([powerlawdist()], dtype = torch.float32)
        #30 degree angle of incidence
        theta = torch.tensor([np.random.uniform(0, theta_max)], dtype = torch.float32)

        phi = torch.tensor([np.random.uniform(-np.pi, np.pi)], dtype = torch.float32)

    #Evalute the counts in the tanks
    Ne, Nm, Te, Tm = GetCounts(energy, theta, phi, X0, Y0, x, y, shower_type)

    return Ne, Nm, Te, Tm, X0, Y0, energy, theta, phi, shower_type


def GetCounts(TrueE, TrueTh, TruePhi, TrueX0, TrueY0, x, y, shower_type):
    #Initialize Counts and Times
    Nm_list = []
    Ne_list = []
    Tm_list = []
    Te_list = []

    for idx in range(Nunits):
        #At this part we are evaluating the number of particles detected in each unit
        R = EffectiveDistance(x[idx], y[idx], TrueX0, TrueY0, TrueTh, TruePhi)
        ct = torch.cos(TrueTh)

        #Evaluate the Content
        if shower_type == "g":
            m0 = ShowerContent(TrueE, TrueTh, R, "mg")
            e0 = ShowerContent(TrueE, TrueTh, R, "eg")
        elif shower_type == "h":
            m0 = ShowerContent(TrueE, TrueTh, R, "mp")
            e0 = ShowerContent(TrueE, TrueTh, R, "ep")

        mb = fluxB_m
        eb = fluxB_e

        nms = SmearN(m0 * ct)
        nes = SmearN(e0 * ct)

        nmb = SmearN(fluxB_m)
        neb = SmearN(fluxB_e)

        Nm_i = nms + nmb
        Ne_i = nes + neb

        Nm_list.append(Nm_i)
        Ne_list.append(Ne_i)

        #Handling the timing generation
        et = EffectiveTime(x[idx], y[idx], TrueX0, TrueY0, TrueTh, TruePhi) #expected time of arrival in the unit

        # Conditional times (avoid .item())
        if Ne_i > 0:
            TAe_m, TAe_s = TimeAverage(et, neb, nes)
            Te_list.append(TAe_m + torch.randn_like(TAe_m) * TAe_s)
        else:
            Te_list.append(torch.tensor([0.0]))

        if Nm_i > 0:
            TAm_m, TAm_s = TimeAverage(et, nmb, nms)
            Tm_list.append(TAm_m + torch.randn_like(TAm_m) * TAm_s)
        else:
            Tm_list.append(torch.tensor([0.0]))

    # Stack to form tensors
    Nm = torch.stack(Nm_list)
    Ne = torch.stack(Ne_list)
    Tm = torch.stack(Tm_list)
    Te = torch.stack(Te_list)

    return Ne, Nm, Te, Tm

def EffectiveDistance(xd, yd, x0, y0, th, ph):
    dx = xd - x0
    dy = yd - y0
    t = torch.sin(th) * torch.cos(ph) * dx + torch.sin(th) * torch.sin(ph) * dy
    r = dx ** 2 + dy ** 2 - t ** 2

    r = torch.where(r > 0, torch.sqrt(r), torch.zeros_like(r))  # or any fallback like r=r
    r = torch.clamp(r, min = R_min)

    return r

def EffectiveTime(xd, yd, x0, y0, th, ph):
    et = ((xd - x0) * torch.sin(th) * torch.cos(ph) + (yd - y0) * torch.sin(th) * torch.sin(ph)) / c0

    return et


def SmearN(flux):
    gate = torch.sigmoid(10 * (flux - 0.1))
    noise = torch.randn_like(flux)  # standard normal noise
    noisy = flux + RelResCounts * flux * noise  # reparameterized: mean + std * noise

    return gate * noisy


def TimeAverage(T, Nb, Ns):
    noise = torch.rand_like(T) - .5

    if Nb <= 1:
        STbgr = IntegrationWindow / sqrt12
        AvTbgr = T + noise * STbgr

    elif Nb <= 2:
        STbgr = IntegrationWindow * .2041
        AvTbgr = T + noise * STbgr

    elif Nb <= 3:
        STbgr = IntegrationWindow * .166666
        AvTbgr = T + noise * STbgr

    elif Nb <= 4:
        STbgr = IntegrationWindow * .1445
        AvTbgr = T + noise * STbgr

    else:
        STbgr = IntegrationWindow * .11
        AvTbgr = torch.normal(T, STbgr)

        while (AvTbgr - T > .5 * IntegrationWindow):
            AvTbgr = torch.normal(T, STbgr)

    STsig = sigma_time

    if Ns >= 2:
        STsig = sigma_time / torch.sqrt(Ns - 1)

    AvTsig = T + torch.randn_like(T) * STsig

    if Nb == 0 and Ns == 0:
        mean = T
        std = IntegrationWindow / sqrt12

    elif Nb == 0:
        mean = AvTsig
        std = STsig

    elif Ns == 0:
        mean = AvTbgr
        std = STbgr

    else:
        w_bgr = 1.0 / (STbgr ** 2)
        w_sig = 1.0 / (STsig ** 2)
        sum_weights = w_bgr + w_sig

        mean = (AvTsig * w_sig + AvTbgr * w_bgr) / sum_weights
        std = torch.sqrt(1.0 / sum_weights)

    return mean, std


def powerlawdist(E_min = 0.1, E_max = 10, index = 2.7):
    # Inverse transform sampling
    r = np.random.rand()
    exponent = 1.0 - index
    Emin_pow = E_min**exponent
    Emax_pow = E_max**exponent
    energies = (Emin_pow + r * (Emax_pow - Emin_pow)) ** (1.0 / exponent)

    return energies

def hg_dist():
    r = np.random.rand()

    if r <= 1e-4:
        shower = "g"
    else:
        shower = "h"

    return shower


def symmetry_loss(x, y, n_symmetry = 3, center = (0.0, 0.0)):
    #Penalizes deviation from n-fold rotational symmetry without relying on grouping.
    #Matches each rotated point to its nearest neighbor in the original set.
    x_centered = x - center[0]
    y_centered = y - center[1]
    coords = torch.stack([x_centered, y_centered], dim=1)  # (N, 2)

    sym_loss = 0.0
    for i in range(1, n_symmetry):
        theta = 2 * np.pi * i / n_symmetry
        R = torch.tensor([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ], dtype=coords.dtype, device=coords.device)

        rotated = coords @ R.T  # (N, 2)

        # Compute pairwise distances between rotated and original coords
        dists = torch.cdist(rotated, coords, p=2)  # (N, N)

        # Find nearest neighbor for each rotated point
        min_dists, _ = dists.min(dim=1)

        sym_loss += min_dists.mean()

    return sym_loss / (n_symmetry - 1)


def shower_converter(s):
    if s == "h":
        b = 1.0
    elif s == "g":
        b = .0
    else:
        warnings.warn("Shower type is not defined!")

    return b

def shower_identification(v):
    if v >= .5:
        t = "h"
    elif 0 <= v and v < .5:
        t = "g"
    else:
        warnings.warn("Shower type is not defined!")

    return t

print('all libraries were loaded')





# NEURAL NETWORK 
'''
In this section we are creating our neural network, we will train it with some simulated EAS. the idea is to feed the network with detector positions, with detected number of particles, and time of arrival. Then we neural network output will be energy of the shower, angles, and shower core (X, Y)

We have to determine the number of hidden layers and number of neurons by hand, as there is no optimization regarding these parameters.

We will keep learning rate small, ie. lr = 1e-5, otherwise NN tends to overfit

It might be argued that for the first part there is no reason to keep positions of the detectors as inputs in our network, but that part will be useful when we need to train our NN again for new positions.
'''
class Reconstruction(nn.Module):
    def __init__(self, input_features = 6, num_detectors = 90, hidden_lay1 = 512, hidden_lay2 = 256,
                 hidden_lay3 = 128, hidden_lay4 = 64, hidden_lay5 = 32, output_dim = 7):
        super(Reconstruction, self).__init__()
        self.num_detectors = num_detectors
        self.input_features = input_features

        #We have to flatten the input, since we are using fully connected neural network
        self.L1 = nn.Linear(num_detectors * input_features, hidden_lay1)
        self.L2 = nn.Linear(hidden_lay1, hidden_lay2)
        self.L3 = nn.Linear(hidden_lay2, hidden_lay3)
        self.L4 = nn.Linear(hidden_lay3, hidden_lay4)
        self.L5 = nn.Linear(hidden_lay4, hidden_lay5)
        self.L6 = nn.Linear(hidden_lay5, output_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1) #This way we drop a fraction of the neurons randomly at iteration,
        #So that our network won't rely on some specific network path

        #Output_dim = 6 means: we will have an output containing (X0, Y0, E0, Theta0, Phi0, Shower_type)

    def forward(self, x):
        out = self.L1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.L2(out)
        out = self.relu(out)
        out = self.L3(out)
        out = self.relu(out)
        out = self.L4(out)
        out = self.relu(out)
        out = self.L5(out)
        out = self.relu(out)
        out = self.L6(out)

        class_out = self.sig(out[:, 6:7])
        param_out = self.tanh(out[:, 0:4])
        phi_out = out[:, 4:6]

        return torch.cat([param_out, phi_out, class_out], dim = 1)

#These functions are needed to normalize the labels and denormalize the outputs, otherwise the scales change too much that
#NN training will fail
def NormalizeLabels(E, theta, phi):
    E_norm = 2 * (E - .1) / (10 - .1) - 1
    theta_norm = 2 * theta / (theta_max) - 1
    sin = torch.sin(phi)
    cos = torch.cos(phi)

    return E_norm, theta_norm, sin, cos

def DenormalizeLabels(E_norm, theta_norm, sin, cos):
    E = 0.1 + (E_norm + 1) * (10 - 0.1) / 2
    theta = (theta_norm + 1) * theta_max / 2
    phi = torch.atan2(sin, cos)

    return E, theta, phi


#We use this class in the case of early stop
#If our network stops improving after some time there is no need to continue

class EarlyStopping:
    def __init__(self, patience = 30, min_delta = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True





# UTILITIES and OPTIMIZATION
'''
These functions are defined to optimize the unit positions maximizing the value of utility functions
'''
def reconstructability(events):
    layout_threshold = 5. #We accept the arrays detecting >= 5 particles
    tau_layout = 5.
    reconstruct_threshold = 3.
    tau_reconstruct = 5.

    soft_detect = torch.sigmoid(tau_layout * (events - layout_threshold))
    n = torch.sum(soft_detect, dim = 1)

    r = torch.sigmoid(tau_reconstruct * (n - reconstruct_threshold))

    return r

#I will try to smooth the function and make it differentiable for PyTorch, the code is below, I am not 100% sure if this
#code translates exactly what I want to do, but we can check

def U_PR(r):
    u = torch.sqrt(torch.sum(r) + 1e-6) #1e-6 is for stability

    return u



def U_E(E_preds, E_trues, r):
    u = torch.sum(r / ((E_preds - E_trues) ** 2 + .01))

    return u



def U_TH(Th_preds, Th_trues, r):
    u = torch.sum(r / ((Th_preds - Th_trues) ** 2 + .00001))

    return u



def barycentric_coords(P, A, B, C):
    """
    Compute barycentric coordinates for each point P with respect to triangle ABC.
    P: Tensor of shape (N, 2)
    A, B, C: Tensors of shape (2,)
    """
    v0 = C - A
    v1 = B - A
    v2 = P - A

    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)

    denom = d00 * d11 - d01 * d01 + 1e-8
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    return u, v



def project_to_triangle(x, y):
    """
    Projects each (x[i], y[i]) point inside the triangle defined by:
    A = (-3200, 2000), B = (1800, 2000), C = (1800, -3600)
    x, y: tensors of shape (N,)
    Returns: projected x and y, tensors of shape (N,)
    """
    A = torch.tensor([-3800.0, 1500.0], device=x.device)
    B = torch.tensor([1200.0, 1500.0], device=x.device)
    C = torch.tensor([1200.0, -4100.0], device=x.device)

    P = torch.stack([x, y], dim=1)  # (N, 2)

    # Compute barycentric coordinates
    u, v = barycentric_coords(P, A, B, C)

    # Determine which points are inside the triangle
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)

    # Clip to triangle: u, v ∈ [0, 1], u + v ≤ 1
    u_clipped = torch.clamp(u, 0.0, 1.0)
    v_clipped = torch.clamp(v, 0.0, 1.0)
    uv_sum = u_clipped + v_clipped
    over = uv_sum > 1.0
    u_clipped[over] = u_clipped[over] / uv_sum[over]
    v_clipped[over] = v_clipped[over] / uv_sum[over]

    v0 = C - A
    v1 = B - A
    P_proj = A + u_clipped.unsqueeze(1) * v0 + v_clipped.unsqueeze(1) * v1

    # If already inside, keep P. Otherwise, use projection.
    final_P = torch.where(inside.unsqueeze(1), P, P_proj)

    return final_P[:, 0], final_P[:, 1]  # x_proj, y_proj



class LearnableXY(torch.nn.Module):
    def __init__(self, x_init, y_init):
        super().__init__()
        self.x = torch.nn.Parameter(x_init)
        self.y = torch.nn.Parameter(y_init)

    def forward(self):
        return self.x, self.y




def push_apart(module, min_dist = 2 * TankRadius):
    x, y = module()  # Correctly calls forward()
    coords = torch.stack([x, y], dim=1)  # shape (N, 2)

    with torch.no_grad():
        for i in range(coords.shape[0]):
            diffs = coords[i] - coords
            dists = torch.norm(diffs, dim=1)
            mask = (dists < min_dist) & (dists > 0)

            for j in torch.where(mask)[0]:
                direction = diffs[j] / dists[j]
                displacement = 0.5 * (min_dist - dists[j]) * direction
                coords[i] += displacement
                coords[j] -= displacement

        # Update learnable parameters in-place
        module.x.data.copy_(coords[:, 0])
        module.y.data.copy_(coords[:, 1])
