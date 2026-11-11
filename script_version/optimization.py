import time
import argparse
import sys
import functions
functions.SWGOopt = True
from functions import *
t0 = time.time()

'''
This is part where we start moving our arrays according to Utility Function.
You can check the utilities that we use above and see that it is > 0
So we will use -U as a "loss" function and minimize it, so that we can maximize our value
'''
parser = argparse.ArgumentParser()
parser.add_argument('--model_path',type=str)
args = parser.parse_args()
set_name = Path(args.model_path).stem

model = Reconstruction()
path = Path('./checkpoint.pth')

if path.exists():
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Updated Weights are loaded")

else:
    model.load_state_dict(torch.load(args.model_path))
    print("Initial Weights are loaded")


#e = 0
#for i in range(1000):
#    p_layout = Path(f"./Python_Layouts/Layout_{i + 1}.txt")
#    
#    if p_layout.exists():
#        data = np.loadtxt(p_layout)
#        
#        x = torch.tensor(data[:, 0], dtype = torch.float32)
#        y = torch.tensor(data[:, 1], dtype = torch.float32)
#        e = i + 1
#
#if e > 0:
#    print(f"Updated Layout {e} is initialized")
#
#else:
#    print("First Layout is initialized")

data = np.loadtxt("./Python_Layouts/Layout_1000.txt")
x = torch.tensor(data[:, 0], dtype = torch.float32)
y = torch.tensor(data[:, 1], dtype = torch.float32)
xy_module = LearnableXY(x,y)

#Let's visualize the initial Layout
#plt.figure(figsize=(7, 7))
#plt.scatter(x.numpy(), y.numpy(), color='blue', marker='o', s=30)

#plt.xlabel("x [m]")
#plt.ylabel("y [m]")
#plt.title("Initial Detector Layout")
#plt.axis("equal")
#plt.grid(True)
#plt.show()

#Now we do the optimization step
optimizer = torch.optim.SGD(xy_module.parameters(), lr = 5, momentum = .9)

if path.exists():
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    num_epoch = checkpoint.get("epoch") + 1
    loss = checkpoint.get("loss")

    print(f"Optimizer is initialized from the last epoch {num_epoch}")

else:
    num_epoch = 0
    print("Optimizer is initialized")

max_grad = 10.

Nbatch = 500

if num_epoch < 20:
    Nfinetune = 1500

elif num_epoch >= 20:
    Nfinetune = 3000

U_vals = []
U_pr_vals = []
U_e_vals = []
U_th_vals = []



for epoch in range(num_epoch, num_epoch + 10):
    x1, y1 = xy_module()

    #Batch Generation:
    Ne_list = []
    Nm_list = []
    Te_list = []
    Tm_list = []
    labels_batch = torch.zeros((Nbatch, 5))

    for i in range(Nbatch):
        Ne, Nm, Te, Tm, x0, y0, E, theta, phi, shower_type = GenerateShowers(x1, y1)

        Ne_list.append(Ne)
        Nm_list.append(Nm)
        Te_list.append(Te)
        Tm_list.append(Tm)
        labels_batch[i] = torch.tensor([x0, y0, E, theta, phi], dtype = torch.float32)

    x = x1.unsqueeze(0).repeat(Nbatch, 1)
    y = y1.unsqueeze(0).repeat(Nbatch, 1)

    events_ne_batch = torch.stack(Ne_list).squeeze(-1)  # shape: (Nbatch, Nunits)
    events_nm_batch = torch.stack(Nm_list).squeeze(-1)
    events_te_batch = torch.stack(Te_list).squeeze(-1)
    events_tm_batch = torch.stack(Tm_list).squeeze(-1)

    events_batch = torch.stack((x, y, events_ne_batch, events_nm_batch, events_te_batch, events_tm_batch), dim = 2)

    model.eval()

    preds_batch = model(events_batch.view(Nbatch, -1))

    #Denormalize the predictions
    preds_x = preds_batch[:, 0] * 5000
    preds_y = preds_batch[:, 1] * 5000
    preds_e, preds_th, preds_phi = DenormalizeLabels(preds_batch[:, 2], preds_batch[:, 3], preds_batch[:, 4], preds_batch[:,5])

    #Compute the reconstructability score for each event:
    r_score = reconstructability(events_batch[:, :, 2])
    density = Nbatch / ((labels_batch[:, 0].max() - labels_batch[:, 0].min()) *
                        (labels_batch[:, 1].max() - labels_batch[:, 1].min()))

    #Compute Utility:
    U = 1e-2 * U_TH(preds_th, labels_batch[:, 3], r_score) + U_E(preds_e, labels_batch[:, 2], r_score) + U_PR(r_score) / torch.sqrt(density)

    #Save the utility values to plot
    U_vals.append(U.item())
    U_pr_vals.append((U_PR(r_score) / torch.sqrt(density)).item())
    U_e_vals.append(U_E(preds_e, labels_batch[:, 2], r_score).item())
    U_th_vals.append(1e-2 * U_TH(preds_th, labels_batch[:, 3], r_score).item())

    #sym_loss = symmetry_loss(x1, y1, n_symmetry = 3)
    print(f"Utility: {U:.2f}")

    #Use utility as - Loss, we have a penalty term to make configuration symmetric
    Loss = - U #+ lambda_sym * sym_loss

    Loss.backward()

    torch.nn.utils.clip_grad_norm_(xy_module.parameters(), max_norm = max_grad)

    optimizer.step()

    with torch.no_grad():
        push_apart(xy_module)

    #This part is done to keep the detectors inside the site
    """
    with torch.no_grad():
        x_proj, y_proj = project_to_triangle(x1.view(-1), y1.view(-1))
        x1.copy_(x_proj)
        y1.copy_(y_proj)
    """

    optimizer.zero_grad()

    #Now we need to generate new events that with the new layout so that we can fine tune our NN when it's necessary
    if (epoch + 1) % 5 == 0:
        print(f"Fine Tune at epoch {epoch + 1}")

        finetune_events = torch.zeros((Nfinetune, Nunits, 6))
        finetune_trues = torch.zeros((Nfinetune, 7))

        with torch.no_grad():
            x2, y2 = xy_module()
            x = x2.detach()
            y = y2.detach()

            #Generate events to fine tune the network
            for i in range(Nfinetune):
                Ne, Nm, Te, Tm, x0, y0, E, theta, phi, shower_type = GenerateShowers(x, y)

                x0 /= 5000
                y0 /= 5000
                E_norm, theta_norm, sin, cos = NormalizeLabels(E, theta, phi) ### why this shower generation needs normalization and previous one not?
                shower_value = shower_converter(shower_type)

                finetune_events[i] = torch.tensor(np.column_stack((x, y, Ne, Nm, Te, Tm)))
                finetune_trues[i] = torch.tensor([x0, y0, E_norm, theta_norm, sin, cos, shower_value], dtype = torch.float32)

        ReconstructionNN = model
        ReconstructionNN.train()
        criterion_cont = nn.MSELoss() 
        criterion_disc = nn.BCELoss()
        criterion_type = nn.BCELoss()
        optimizerNN = torch.optim.Adam(ReconstructionNN.parameters(), lr = 1e-4)

        dataset = TensorDataset(finetune_events, finetune_trues)
        dataloader = DataLoader(dataset, batch_size = 32, shuffle = True, drop_last = True, num_workers = 4)
        
        print('finetune_events', dataset.tensors[0].shape)  # finetune_events
        print('finetune_trues', dataset.tensors[1].shape)  # finetune_trues

        for j in range(10): ### why range(5)? what is 5 due to?
            
            progress = j / 10.0  # normalize to [0,1]
            wcore = 50 * (1 - 0.9 * progress)   # from 50 → ~5
            wcont = 0.05 + 0.45 * progress      # from 0.05 → ~0.5
            wdisc = 1

            for ft_batch, ft_trues in dataloader:
                
                batch_size = ft_batch.size(0)
                train_x = ft_batch.view(batch_size, -1) #inputs from new batch
                train_y = ft_trues.view(batch_size, -1) #real values from new batch

                #Train the network
                outputs = ReconstructionNN(train_x) #labels or outputs from new batch dedicated to NN learning/adjusting

                #radial loss
                dx = outputs[:,0]-train_y[:,0]
                dy = outputs[:,1]-train_y[:,1]
                core_loss = torch.mean(torch.sqrt(dx**2+dy**2))

                #continuous loss (E, theta, phi, N)
                cont_loss = criterion_cont(outputs[:,2:6], train_y[:,2:6])
                
                #discrete loss (gamma/hadron class)
                disc_loss = criterion_type(outputs[:,6].unsqueeze(1), train_y[:,6].unsqueeze(1))

                WL = {
                    "core": wcore * core_loss,
                    "cont": wcont * cont_loss,
                    "disc": wdisc * disc_loss
                }

                loss = WL['core'] + WL['cont'] + WL['disc']
                #lossT = criterion(outputs, train_y)
                #loss = .05 * criterion_cont(outputs[:, :6], train_y[:, :6]) + criterion_disc(outputs[:, 6].unsqueeze(1)
                #                                                               , train_y[:, 6].unsqueeze(1))
                
                loss.backward()
                optimizerNN.step()

                optimizerNN.zero_grad()

       
    #I will save the layouts and weights here so that I can stop whenever and continue later
    torch.save({"epoch": epoch, "loss": Loss, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, "checkpoint.pth")

    np.savetxt(f"./Python_Layouts/Layout_{epoch + 1}.txt", np.column_stack((x1.detach().numpy(), y1.detach().numpy())))



# Plot optimized array

a, b = Layouts()
x_final, y_final = xy_module()

plt.figure(figsize = [6, 6])
plt.scatter(x_final.detach().numpy(), y_final.detach().numpy(), color = "red", alpha = .8, label = "Final Layout")
plt.scatter(a, b, color = "blue", alpha = .3, label = "Initial Layout")
#plt.plot([-3800, 1200, 1200, -3800], [1500, 1500, -4100, 1500], "k--", label = "SWGO Site")

#Add Shower Simulation Area
#ax = plt.gca()
#rect = patches.Rectangle((-3000, -3000), 6000, 6000, linewidth=1.5, edgecolor='green', facecolor='green',
                         #alpha=0.1, label='Shower Simulation Area')
#ax.add_patch(rect)
plt.axis('equal')
plt.grid()
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Position of the detectors")
plt.legend()
plt.xlim((-3000, 3000))
plt.ylim((-3000, 3000))
plt.savefig(f"optimization_layout_{set_name}.png")
plt.close()


# Utility plot

utility_path = Path(f"./Python_Layouts/utility_{set_name}.txt")

if utility_path.exists():
    data = np.loadtxt(utility_path)
    u_t = data
    #u_p = data[:, 1]
    #u_e = data[:, 2]
    #u_a = data[:, 3]

    u_t = np.append(u_t, np.array(U_vals).ravel())
    #u_p = np.append(u_p, np.array(U_pr_vals).ravel())
    #u_e = np.append(u_e, np.array(U_e_vals).ravel())
    #u_a = np.append(u_a, np.array(U_th_vals).ravel())

    #data = np.column_stack((u_t, u_p, u_e, u_a))

    np.savetxt(utility_path, u_t)

else:
    u_t = U_vals
    #u_p = U_pr_vals
    #u_e = U_e_vals
    #u_a = U_th_vals

    #data = np.column_stack((u_t, u_p, u_e, u_a))
    np.savetxt(utility_path, u_t)

mean_ut = [np.mean(u_t[i - 4: i]) for i in range(4, len(u_t))]

plt.figure(figsize=[6,6])
plt.plot(u_t, color = "black", linestyle = "none", marker = "o", linewidth = .5, label = "Total Utility")
plt.plot(np.arange(4, len(u_t)), mean_ut, linestyle = "-", linewidth = 2, label = "Mean Utility")
#plt.plot(u_p, color = "orange", linestyle = "-.", label = "Reconstructability Utility")
#plt.plot(u_e, color = "green", linestyle = "-.", label = "Energy Utility")
#plt.plot(u_a, color = "purple", linestyle = "-.", label = "Angle Utility")
plt.legend()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Utility Score")
plt.ylim((0, np.max(u_t) + 1000))
plt.title("Utility Values per Optimization Step")
plt.savefig(f"utility_{set_name}.png")

print("--- %s seconds ---" % (time.time() - t0))
