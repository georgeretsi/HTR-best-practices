import argparse
from omegaconf import OmegaConf

import sys
import os
import tqdm
import numpy as np
import torch
import torch.nn as nn

from models import HTRNet
from utils.preprocessing import load_image, preprocess


class HTREval(nn.Module):
    def __init__(self, config):
        super(HTREval, self).__init__()
        self.config = config

        self.prepare_dataloaders()
        self.prepare_net()

    def prepare_dataloaders(self):

        config = self.config

        # prepare datset loader
        dataset_folder = config.data.path
        # load classes from the training set saved in the data folder
        classes = np.load(os.path.join(dataset_folder, 'classes.npy'))

        # create dictionaries for character to index and index to character 
        # 0 index is reserved for CTC blank
        cdict = {c:(i+1) for i,c in enumerate(classes)}
        icdict = {(i+1):c for i,c in enumerate(classes)}

        self.classes = {
            'classes': classes,
            'c2i': cdict,
            'i2c': icdict
        }

    def prepare_net(self):

        config = self.config

        device = config.device

        print('Preparing Net - Architectural elements:')
        print(config.arch)

        classes = self.classes['classes']

        net = HTRNet(config.arch, len(classes) + 1)
        
        if config.resume is not None:
            print('resuming from checkpoint: {}'.format(config.resume))
            load_dict = torch.load(config.resume)
            load_status = net.load_state_dict(load_dict, strict=True)
            print(load_status)
        net.to(device)

        # print number of parameters
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Number of parameters: {}'.format(n_params))

        self.net = net

    def decode(self, tdec, tdict, blank_id=0):
        
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([tdict[t] for t in tt if t != blank_id])
        
        return dec_transcr
                
    def sample_decoding(self, img_path):

        # get a random image from the test set
        img = load_image(img_path)
        img = preprocess(img, (self.config.preproc.image_height, self.config.preproc.image_width))
        img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

        img = img.to(self.config.device)

        self.net.eval()
        with torch.no_grad():
            tst_o = self.net(img)
            if self.config.arch.head_type == 'both':
                tst_o = tst_o[0]

        self.net.train()

        tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        # remove duplicates
        dec_transcr = self.decode(tdec, self.classes['i2c'])

        print('predicted:: ' + dec_transcr.strip())


def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    img_path = sys.argv[-1]

    sys.argv = [sys.argv[0]] + sys.argv[2:-1] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf, img_path


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config, img_path = parse_args()
    max_epochs = config.train.num_epochs

    htr_eval = HTREval(config)

    htr_eval.sample_decoding(img_path)


    