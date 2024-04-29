import argparse
import logging
import os
from glob import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import METANAFNet
from msssimLoss import MSSSIM
import scipy.io as scio

dir_input = './meta_test_dataset/seismic_task/test/' 
dir_output='./meta_test_dataset/seismic_task/output_'

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

def predict_input(net,
                full_input,full_label,
                device):

    net.eval()
    
    criterion = nn.MSELoss()
    criterion2 = MSSSIM()

    inputs = torch.from_numpy(full_input).type(torch.FloatTensor)
    labels = torch.from_numpy(full_label).type(torch.FloatTensor)
    inputs = inputs.unsqueeze(0).unsqueeze(1)  
    inputs = inputs.to(device = device, dtype = torch.float32)
    labels = labels.unsqueeze(0).unsqueeze(1)
    labels = labels.to(device = device, dtype = torch.float32)

    with torch.no_grad():
        output = net(inputs)
        test_loss = criterion(output, labels)
        test_msssim = criterion2(output, labels)
        probs = output.squeeze(0)
        
        outputs_pred = probs.squeeze().cpu().numpy()

    return outputs_pred, test_loss, test_msssim


def get_args():
    parser = argparse.ArgumentParser(description='Predict output_preds from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    args = get_args()
    in_files = get_ids(dir_input)
    model_id = os.path.splitext(args.model)[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]
    width = 32

    net = METANAFNet(in_channels=1, width=width, middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)
    net.to(device = device)
    net.load_state_dict(torch.load(args.model, map_location = device))
    print(f'Model loaded from {args.model}')

    dir_output_epoch = dir_output+args.model[17:-4]+'/'
    try:
        os.makedirs(dir_output_epoch)
    except OSError:
        pass

    logging.info("Loading model {}".format(args.model))

    criterion = nn.MSELoss()
    
    print('------ Test starting -------')
    for i, fn in enumerate(in_files):
        input_file = glob(dir_input + fn + '.*')
        logging.info("\nPredicting seismic data {} ...".format(input_file))
        dict = scio.loadmat(input_file[0])
        input = dict['input']
        label = dict['label']
       
        output_pred, loss, msssim = predict_input(net=net,
                           full_input=input,
                           full_label=label,
                           device=device)

        print("\nPredicting seismic data {} have done...".format(fn))
        
        scio.savemat(dir_output_epoch + fn + "_OUT.mat" ,{'predict': output_pred, 'loss':loss.item(), 'msssim':msssim.item()})
        logging.info("Predict result saved to {}".format(fn + "_OUT.mat"))
    print('------ Test completed successfully -------')
