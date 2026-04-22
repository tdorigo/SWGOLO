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
import time
from pathlib import Path
from matplotlib import colors
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from torch.utils.data import TensorDataset, DataLoader
#from functions import Layouts, Reconstruction, NormalizeLabels, DenormalizeLabels, EarlyStopping
from functions import *
t0 = time.time()
x, y = Layouts()
x = torch.tensor(x, dtype = torch.float32)
y = torch.tensor(y, dtype = torch.float32)


# TEST DATA GENERATION. Number of showers in the test set = 200k #

inputs = torch.zeros((Nevents, Nunits, 6))
labels = torch.zeros((Nevents, 7))

for i in range(Nevents):
    Ne, Nm, Te, Tm, x0, y0, E, theta, phi, shower_type = GenerateShowers(x, y)

    #Normalize the Labels:
    E_norm, theta_norm, sin, cos = NormalizeLabels(E, theta, phi)
    x0 = x0 / 5000 #[km]
    y0 = y0 / 5000 #[km]
    shower_value = shower_converter(shower_type)

    input_vector = np.column_stack((x, y, Ne, Nm, Te, Tm))

    inputs[i] = torch.tensor(input_vector, dtype = torch.float32)
    labels[i] = torch.tensor([x0, y0, E_norm, theta_norm, sin, cos, shower_value], dtype = torch.float32)

    if (i + 1) % 3000 == 0:
        print(f"Shower generation is {int((i + 1) / 300)}% done")

torch.save(inputs, "./NN_Files/inputs.pt")
torch.save(labels, "./NN_Files/labels.pt")



# VALIDATION SET DATA GENERATION Number of events in the validation set = 20k, i.e 10% of the training set #

inputs_val = torch.zeros((Nvalidation, Nunits, 6))
labels_val = torch.zeros((Nvalidation, 7))

for i in range(Nvalidation):
    Ne, Nm, Te, Tm, x0, y0, E, theta, phi, shower_type = GenerateShowers(x, y)

    #Normalize the Labels:
    E_norm, theta_norm, sin, cos = NormalizeLabels(E, theta, phi)
    x0 = x0 / 5000 #[km]
    y0 = y0 / 5000 #[km]
    shower_value = shower_converter(shower_type)

    input_vector = np.column_stack((x, y, Ne, Nm, Te, Tm))

    inputs_val[i] = torch.tensor(input_vector, dtype = torch.float32)
    labels_val[i] = torch.tensor([x0, y0, E_norm, theta_norm, sin, cos, shower_value], dtype = torch.float32)

    if (i + 1) % 2000 == 0:
        print(f"Shower generation is {int((i + 1) / 200)}% done")

torch.save(inputs_val, "./NN_Files/inputs_val.pt")
torch.save(labels_val, "./NN_Files/labels_val.pt")
print("--- %s seconds ---" % (time.time() - t0))
print("validation set saved")
