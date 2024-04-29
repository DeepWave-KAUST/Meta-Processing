import  torch.utils.data as data
import  os
import  os.path
import torch
import re
from glob import glob
from os.path import splitext
from os import listdir
import scipy.io as scio
import numpy as np

class Basicdataset(data.Dataset):
    def __init__(self, dir):
        self.dir = dir

        self.ids = strsort([splitext(file)[0] for file in listdir(self.dir)
                    if not file.startswith('.')])

    def __getitem__(self, index):
        idx_file = self.ids[index]

        file = glob(self.dir + idx_file +  '.*')

        dict = scio.loadmat(file[0])
        input = dict['input']
        label = dict['label']

        return {
            'input': torch.from_numpy(input).unsqueeze(0).type(torch.FloatTensor),
            'label': torch.from_numpy(label).unsqueeze(0).type(torch.FloatTensor)
        }

    def __len__(self):
        return len(self.ids)

def sort_key(s):
    tail = s.split('\\')[-1]
    c = re.findall('\d+', tail)[0]
    return int(c)
 
def strsort(alist):
    alist.sort(key=sort_key)
    return alist
