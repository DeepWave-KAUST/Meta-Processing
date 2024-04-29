import  torch, os
import  numpy as np
import  argparse
import torch.nn as nn
from    model import METANAFNet
from collections import OrderedDict
from msssimLoss import MSSSIM
from dataset import Basicdataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import scipy.io as scio
from eval import eval_net
import re
from glob import glob

'''
Define the dataset for meta-test.
If you download the dataset, you can modify here accordingly.
'''
dir_train = './meta_test_dataset/seismic_task(e.g., vrms, migration, denoise, interpolation,ground_roll)/train/'
dir_test = './meta_test_dataset/seismic_task/test/'
dir_checkpoint = './checkpoints/'     # define the checkpoints file folder
dir_output = './dataset/output/'      # define the test output file folder
dir_load = './meta_checkpoints/meta_initialization.pth'     # define the meta initialization file folder

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    dataset = Basicdataset(dir_train)
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    device = torch.device('cuda')

    print(f'''Starting training:
        Epochs:          {args.epoch}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]
    width = 32

    net = METANAFNet(in_channels=1, width=width, middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)

    net.load_state_dict(torch.load(dir_load, map_location=device))
    print(f'Model loaded from {dir_load}')

    net.to(device=device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98) 

    criterion = nn.MSELoss()
    criterion2 = MSSSIM()

    writer = SummaryWriter(comment=f'LR_{args.lr}_BS_{args.batch_size}_Epoch_{args.epoch}')

    try:
        os.makedirs(dir_checkpoint)
    except OSError:
        pass

    try:
        os.makedirs(dir_output)
    except OSError:
        pass

    global_step = 0
    for epoch in range(args.epoch):
        net.train() 

        epoch_loss = 0
        epoch_mse = 0
        epoch_msssim = 0
        val_epoch_loss = 0
        val_epoch_msssim = 0
        batch_train_idx = 0
        batch_val_idx = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epoch}', unit='input') as pbar:
            for batch in train_loader:
                batch_train_idx = batch_train_idx + 1
                inputs = batch['input']
                labels = batch['label']

                inputs = inputs.to(device = device, dtype = torch.float32)
                label_type = torch.float32
                labels = labels.to(device = device, dtype = label_type)
                outputs_pred = net(inputs)

                loss1 = criterion(outputs_pred, labels) 
                loss2 = 1 - criterion2(outputs_pred, labels)
                loss = loss1 + loss2 

                epoch_loss += loss.item() 
                epoch_mse += loss1
                epoch_msssim += loss2.item() 

                pbar.set_postfix(**{'loss (batch)': loss1.item(), 'msssim (batch)': (1-loss2).item()})

                writer.add_scalar('Loss_iter/train_loss', loss.item(), global_step)
                writer.add_scalar('Loss_iter/train_loss1', loss1.item(), global_step)
                writer.add_scalar('Loss_iter/train_loss2', loss2.item(), global_step)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                pbar.update(inputs.shape[0])

                global_step += 1

                if batch_train_idx % (n_train // (2 * args.batch_size)) == 0: 
                    batch_val_idx += 1

                    val_loss, val_msssim = eval_net(net, val_loader, device)

                    val_epoch_loss += val_loss
                    val_epoch_msssim += val_msssim

                    print('\nValidation Loss: {}'.format(val_loss))
                    print('Validation MSSSIM: {}'.format(val_msssim))

                    writer.add_scalar('Loss_iter/val_loss', val_loss, global_step)
                    writer.add_scalar('Loss_iter/val_msssim', val_msssim, global_step)

            epoch_loss = epoch_loss/batch_train_idx
            epoch_mse = epoch_mse/batch_train_idx
            epoch_msssim = epoch_msssim/batch_train_idx
            val_epoch_loss = val_epoch_loss/batch_val_idx
            val_epoch_msssim = val_epoch_msssim/batch_val_idx

            writer.add_scalar('Loss_epoch/train_epoch_loss', epoch_loss, epoch)
            writer.add_scalar('Loss_epoch/train_epoch_mse', epoch_mse, epoch)
            writer.add_scalar('Loss_epoch/train_epoch_msssim', epoch_msssim, epoch)
            writer.add_scalar('Loss_epoch/val_epoch_loss', val_epoch_loss, epoch)
            writer.add_scalar('Loss_epoch/val_epoch_msssim', val_epoch_msssim, epoch)

            scheduler.step()

        if (epoch + 1) % 1 == 0:
            torch.save(net.state_dict(), dir_checkpoint+f'CP_epoch{epoch + 1}.pth')
            print(f'Checkpoint {epoch + 1} saved !')

        if (epoch + 1) % 5 == 0:
            print('------------------------- Test starting -------------------------')
            output_pred, file_id, test_loss, test_msssim = predict(dir_test=dir_test, net=net, device=device) 
            scio.savemat(dir_output + file_id + "_OUT_epoch" + str(epoch+1) + ".mat" ,{'predict': output_pred})
            print('Test loss: {}'.format(test_loss))
            print('Test msssim: {}'.format(test_msssim))
            print('------------------ Test completed successfully ------------------')

    writer.close()

def get_ids(dir):
    return strsort([os.path.splitext(file)[0] for file in os.listdir(dir)
                    if not file.startswith('.')]) 

def sort_key(s):
    tail = s.split('\\')[-1]
    c = re.findall('\d+', tail)[0]
    return int(c)
 
def strsort(alist):
    alist.sort(key=sort_key)
    return alist

def predict(dir_test, net, device):

    net.eval()

    criterion = nn.MSELoss()
    criterion2 = MSSSIM()

    in_files = get_ids(dir_test)
    file_id = in_files[0] 
    test_file = dir_test + file_id + '.mat'   
    test_dict = scio.loadmat(test_file)
    test_input = test_dict['input']
    test_label = test_dict['label']
    
    test_input = torch.from_numpy(test_input).type(torch.FloatTensor)
    test_input = test_input.unsqueeze(0).unsqueeze(1) 
    test_input = test_input.to(device = device, dtype = torch.float32)

    test_label = torch.from_numpy(test_label).type(torch.FloatTensor)
    test_label = test_label.unsqueeze(0).unsqueeze(1) 
    test_label = test_label.to(device = device, dtype = torch.float32)

    with torch.no_grad():
        output = net(test_input)
        test_loss = criterion(output, test_label)
        test_msssim = criterion2(output, test_label)
        probs = output.squeeze(0)
        
        outputs_pred = probs.squeeze().cpu().numpy()
    
    return outputs_pred, file_id, test_loss, test_msssim


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=300)
    argparser.add_argument('--batch_size', type=int, help='batch size', default=16)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    argparser.add_argument('--val_percent', type=float, help='val_percent', default=0.1)
    argparser.add_argument('--num_workers', type=int, help='num_workers', default=10)


    args = argparser.parse_args()

    main(args)
