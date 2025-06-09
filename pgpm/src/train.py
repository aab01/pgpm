import os
import numpy as np
import torch as T
from torch import optim
from tqdm import tqdm
from datagenerator import SynthSignalsDataset, get_params_from_json
from model import UNet1SC
from training.train_utils import train_epoch, val_epoch

def train_model(train_ldr, val_ldr, net, device, epochs, batch_size, lr, Loggers, save_cp=True):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.MSELoss()
    lowest_v_L = -1

    for epoch in tqdm(range(epochs)):
        t_loss = train_epoch(epoch, train_ldr, criterion, optimizer, batch_size, scheduler)
        v_loss = val_epoch(epoch, val_ldr, batch_size, criterion)
        scheduler.step()

        if 'WandB' in Loggers:
            wandb.log({'Train Loss': t_loss, 'Val. Loss': v_loss})

        if 'TB' in Loggers:
            writer.add_scalars('Losses', {
                'Train Loss': t_loss,
                'Val. Loss': v_loss,
            }, epoch)

        if save_cp:
            if v_loss < lowest_v_L:
                state_dict = net.state_dict()
                state_dict = net.module.state_dict()
                T.save(state_dict, './savedmodels/model' + f'CP{epoch + 1}.pth')
                print('Checkpoint {} saved !'.format(epoch + 1))
                lowest_v_L = v_loss

    writer.close()
    state_dict = net.state_dict()
    T.save(state_dict, './savedmodels/model' + f'Final.pth')

if __name__ == "__main__":
    # Load parameters and initialize data loaders, model, etc.
    params = get_params_from_json('./Parameters/params.json')
    device = "cuda" if T.cuda.is_available() else "cpu"
    train_ds = SynthSignalsDataset(params, num_samples=N_t, device=device)
    val_ds = SynthSignalsDataset(params, num_samples=N_v, device=device)
    train_ldr = T.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ldr = T.utils.data.DataLoader(val_ds, batch_size=32, shuffle=True)
    net = UNet1SC(n_channels=1, n_classes=6).to(device)

    train_model(train_ldr, val_ldr, net, device, epochs, batch_size=32, lr=lr, Loggers=Loggers)