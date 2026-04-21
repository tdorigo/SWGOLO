from functions import *
import argparse
import time
import sys

'''
In this part we are defining our model, which is the Neural network for our reconstruction
We will use Mean Square Error Loss
Optimization model is SGD, to keep consistent with our next steps
'''
t0 = time.time()




# weight loss estimator 
def estimate_weights(WL, model, optimizer, epoch, batch_idx, N=50):
    """
    WL : dict
        Dictionary of weighted losses, e.g.:
        {
            "core": wcore * core_loss,
            "cont": wcont * cont_loss,
            "disc": wdisc * disc_loss
        }

    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    epoch : int
    batch_idx : int
    N : int
        Run diagnostic every N epochs (default=50)

    Returns
    -------
    grad_dict : dict or None
        Dictionary of gradient magnitudes, or None if not evaluated
    """
    # Only evaluate occasionally (first batch of selected epochs)
    if epoch % N != 0 or batch_idx != 0:
        return None

    grad_dict = {}

    for key, loss in WL.items():
        optimizer.zero_grad()

        loss.backward(retain_graph=True)

        grad_sum = 0.0
        count = 0

        for p in model.parameters():
            if p.grad is not None:
                grad_sum += p.grad.abs().mean().item()
                count += 1

        grad_dict[key] = grad_sum / max(count, 1)

    optimizer.zero_grad()

    print(f"[Gradients @ epoch {epoch}] " +
          ", ".join([f"{k}: {v:.3e}" for k, v in grad_dict.items()]))

    return grad_dict


#####################################################

parser = argparse.ArgumentParser()
parser.add_argument('--inputs',type=str)
parser.add_argument('--inputs_val',type=str)
parser.add_argument('--labels',type=str)
parser.add_argument('--labels_val',type=str)
args = parser.parse_args()

inputs = torch.load(args.inputs)
inputs_val = torch.load(args.inputs_val)
labels = torch.load(args.labels)
labels_val = torch.load(args.labels_val)

x, y = Layouts()
x = torch.tensor(x, dtype = torch.float32)
y = torch.tensor(y, dtype = torch.float32)

model = Reconstruction()
criterion_cont = nn.MSELoss()
criterion_type = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size = 512, shuffle = True, drop_last = True, num_workers = 4)


val_losses = []
losses = []

early_stopper = EarlyStopping()
CL = True
for num_epoch in range(200):
    model.train()
    epoch_loss = 0
    total_batch = 0
    
    if not CL:
        wcore = 47.
        wcont = .05
        wdisc = 1.

    elif CL:
        '''
        curriculum learning: The idea is to prioritize the core learning at the beginning to then "leave space" 
        for energy and angle learning (downweighting).
        '''
        print('CL applied')
        #if num_epoch<30:
        #    wcore = 40
        #    wcont = .05
        #    wdisc = 1.
        #elif num_epoch>=30 and num_epoch<70:
        #    wcore = 15.
        #    wcont = .1
        #    wdisc = 1.
        #elif num_epoch>=70:
        #    wcore = 5.
        #    wcont = .3
        #    wdisc = 1.
        
        progress = num_epoch / 100.0  # normalize to [0,1]

        wcore = 50 * (1 - 0.9 * progress)   # from 50 → ~5
        wcont = 0.05 + 0.45 * progress      # from 0.05 → ~0.5
        wdisc = 1

        if num_epoch>=100: #we freeze after epoch end
            wcore = 5
            wcont = .5
            wdisc = 1
        
    
    if (num_epoch + 1) % 10 == 0:
        print(f"Epoch: {num_epoch}")
        print(f"wcore = {wcore:.2f}")
        print(f"wcont = {wcont:.2f}")
        print(f"wdisc = {wdisc:.2f}")

    for batch_inputs, batch_labels in dataloader:
        batch_size = batch_inputs.size(0)

        train_x = batch_inputs.view(batch_size, -1)

        train_y = batch_labels.view(batch_size, -1)

        #Train the network
        outputs = model(train_x)
        
        # we add dedicated radial core reconstruction (x0, y0)
        dx = outputs[:, 0] - train_y[:, 0]
        dy = outputs[:, 1] - train_y[:, 1]
        core_loss = torch.mean(torch.sqrt(dx**2 + dy**2))

        # N, E, angle: continuous loss. Shower_type: discrete loss
        cont_loss = criterion_cont(outputs[:, 2:6], train_y[:, 2:6])
        disc_loss = criterion_type(outputs[:, 6].unsqueeze(1), train_y[:, 6].unsqueeze(1))
        

        WL = {
            "core": wcore * core_loss,
            "cont": wcont * cont_loss,
            "disc": wdisc * disc_loss
        }

        #estimate_weights(WL, model, optimizer, num_epoch, total_batch, N=50)

        loss = WL['core'] + WL['cont'] + WL['disc']
        epoch_loss += loss.item()
        total_batch += 1

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

    #See if our loss in our validation set improves:
    val_size = inputs_val.size(0)

    val_x = inputs_val.view(val_size, -1)
    val_y = labels_val.view(val_size, -1)

    model.eval()

    with torch.no_grad():

        val_output = model(val_x)

        val_dx = val_output[:, 0] - val_y[:, 0]
        val_dy = val_output[:, 1] - val_y[:, 1]
        val_core_loss = torch.mean(torch.sqrt(val_dx**2 + val_dy**2))

        # N, E, angle continuous loss. shower_type discrete loss
        val_cont_loss = criterion_cont(val_output[:, 2:6], val_y[:, 2:6])
        val_disc_loss = criterion_type(val_output[:, 6].unsqueeze(1), val_y[:, 6].unsqueeze(1))

        val_loss = wcore*val_core_loss + wcont*val_cont_loss + wdisc*val_disc_loss

        #val_loss = .01 * criterion_cont(val_output[:, :6], val_y[:, :6]) + criterion_type(val_output[:, 6].unsqueeze(1)
        #                                                                            , val_y[:, 6].unsqueeze(1))

    val_losses.append(val_loss.item())
    early_stopper(val_loss)
    
    #composite metric loss
    #early_stop_metric = (val_core_loss + 0.2 * val_cont_loss + 0.1 * val_disc_loss)
    #val_losses.append(early_stop_metric.item())
    #early_stopper(early_stop_metric)

    if (num_epoch + 1) % 100 == 0:
        print(f"Training is {int((num_epoch + 1) / 10)}% done, with Loss = {val_loss:.2f}")

    losses.append(epoch_loss / total_batch)

    if early_stopper.early_stop:
        print(f"Early stop at epoch {num_epoch + 1}")
        print(f"Validation loss = {val_loss:.2f}")
        print(f"Training loss = {epoch_loss / total_batch:.2f}")
        break

plt.plot(np.arange(1, len(losses) + 1), losses, color = "blue", label = "Training Loss")
plt.plot(np.arange(1, len(val_losses) + 1), val_losses, color = "red", label = "Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig('./outputs/curriculum_learning/loss.png')

torch.save(model.state_dict(), './NN_Files/curriculum_learning.pth')
print('model saved')
print("--- %s seconds ---" % (time.time() - t0))
