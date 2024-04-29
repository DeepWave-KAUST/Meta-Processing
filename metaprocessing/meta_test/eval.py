import torch
from tqdm import tqdm
import torch.nn as nn
from msssimLoss import MSSSIM

def eval_net(net, loader, device):

    net.eval()
  
    n_val = len(loader)          
    loss1 = 0
    loss2 = 0

    criterion1 = nn.MSELoss()
    criterion2 = MSSSIM()
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            inputs, labels = batch['input'], batch['label']
            inputs = inputs.to(device = device, dtype = torch.float32)
            labels = labels.to(device = device, dtype = torch.float32)

            with torch.no_grad():
                outputs_pred = net(inputs)

            loss1 += criterion1(outputs_pred, labels).item()
            loss2 += criterion2(outputs_pred, labels).item()

            pbar.set_postfix(**{'loss1 (batch)': loss1, 'loss2_msssim (batch)': loss2})

            pbar.update(inputs.shape[0])

    net.train()
    return loss1 / n_val, loss2 / n_val