import argparse
import logging

import os
import numpy as np
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


from utils.iam_dataset import IAMDataset
from utils.rimes_dataset import RimesDataset

from config import *

from models import HTRNet

from utils.auxilary_functions import affine_transformation

import torch.nn.functional as F

logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('HTR-Experiment::train')
logger.info('--- Running HTR Training ---')
# argument parsing
parser = argparse.ArgumentParser()
# - train arguments
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3,
                    help='lr')
parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                    help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
parser.add_argument('--display', action='store', type=int, default=100,
                    help='The number of iterations after which to display the loss values. Default: 100')
parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
parser.add_argument('--scheduler', action='store', type=str, default='mstep')
parser.add_argument('--dataset', action='store', type=str, default='IAM')
parser.add_argument('--remove_spaces', action='store_true')
parser.add_argument('--resize', action='store_true')
parser.add_argument('--head_layers', action='store', type=int, default=3)
parser.add_argument('--head_type', action='store', type=str, default='cnn')

args = parser.parse_args()

gpu_id = args.gpu_id
logger.info('###########################################')

# prepare datset loader

logger.info('Loading dataset.')

if args.dataset == 'IAM':
    myDataset = IAMDataset
    dataset_folder = '/usr/share/datasets_ianos'
elif args.dataset == 'RIMES':
    myDataset = RimesDataset
    dataset_folder = '/usr/share/datasets_ianos'
else:
    raise NotImplementedError

aug_transforms = [lambda x: affine_transformation(x, s=.1)]


if args.dataset == 'IAM':
    train_set = myDataset(dataset_folder, 'train', level, fixed_size=fixed_size, transforms=aug_transforms)
    classes = train_set.character_classes
    print('# training lines ' + str(train_set.__len__()))

    val_set = myDataset(dataset_folder, 'val', level, fixed_size=fixed_size, transforms=None)
    print('# validation lines ' + str(val_set.__len__()))

    test_set = myDataset(dataset_folder, 'test', level, fixed_size=fixed_size, transforms=None)
    print('# testing lines ' + str(test_set.__len__()))
elif args.dataset == 'RIMES':
    train_set = myDataset(dataset_folder, 'train', level, fixed_size=fixed_size, character_classes = None, transforms=aug_transforms)
    classes = train_set.character_classes
    print('# training lines ' + str(train_set.__len__()))
    #train_set = myDataset(args.dataset_folder, 'test', level, fixed_size=fixed_size, character_classes = classes, transforms=aug_transforms)
    test_set = myDataset(dataset_folder, 'test', level, fixed_size=fixed_size, character_classes = classes, transforms=None)
    print('# testing lines ' + str(test_set.__len__()))
    val_set = None

classes = '_' + ''.join(classes)

cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}

# augmentation using data sampler
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
if val_set is not None:
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

# load CNN
logger.info('Preparing Net...')

if args.head_layers > 0:
    head_cfg = (head_cfg[0], args.head_layers)

head_type = args.head_type

net = HTRNet(cnn_cfg, head_cfg, len(classes), head=head_type, flattening=flattening, stn=stn)
net.cuda(args.gpu_id)

ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) /batch_size

#ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='mean', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt)

#restart_epochs = 40 #max_epochs // 6

nlr = args.learning_rate

parameters = list(net.parameters())
optimizer = torch.optim.AdamW(parameters, nlr, weight_decay=0.00005)

if 'mstep' in args.scheduler:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*max_epochs), int(.75*max_epochs)])
elif 'cos' in args.scheduler:
    restart_epochs = int(args.scheduler.replace('cos', ''))
    if not isinstance(restart_epochs, int):
        print('define restart epochs as cos40')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, restart_epochs)
else:
    print('not supported scheduler! choose eithe mstep or cos')

def train(epoch):

    net.train()
    closs = []
    for iter_idx, (img, transcr) in enumerate(train_loader):
        optimizer.zero_grad()

        img = Variable(img.cuda(gpu_id))

        with torch.no_grad():
            rids = torch.BoolTensor(torch.bernoulli(.33 * torch.ones(img.size(0))).bool())
            if sum(rids) > 1 :
                img[rids] += (torch.rand(img[rids].size(0)).view(-1, 1, 1, 1) * 1.0 * torch.randn(img[rids].size())).to(img.device)

        img = img.clamp(0,1)

        if head_type == "both":
            output, aux_output = net(img)
        else:
            output = net(img)

        act_lens = torch.IntTensor(img.size(0)*[output.size(0)])
        labels = torch.IntTensor([cdict[c] for c in ''.join(transcr)])
        label_lens = torch.IntTensor([len(t) for t in transcr])

        loss_val = ctc_loss(output.cpu(), labels, act_lens, label_lens)
        closs += [loss_val.item()]

        if head_type == "both":
            loss_val += 0.1 * ctc_loss(aux_output.cpu(), labels, act_lens, label_lens)
      

        loss_val.backward()

        optimizer.step()


        # mean runing errors??
        if iter_idx % args.display == args.display-1:
            logger.info('Epoch %d, Iteration %d: %f', epoch, iter_idx+1, sum(closs)/len(closs))
            closs = []

            net.eval()

            tst_img, tst_transcr = test_set.__getitem__(np.random.randint(test_set.__len__()))
            print('orig:: ' + tst_transcr)
            with torch.no_grad():
                timg = Variable(tst_img.cuda(gpu_id)).unsqueeze(0)

                tst_o = net(timg)
                if head_type == 'both':
                    tst_o = tst_o[0]

                tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
                tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
                print('gdec:: ' + ''.join([icdict[t] for t in tt]).replace('_', ''))

            net.train()

    if len(closs) > 0 :
        logger.info('Epoch %d, Iteration %d: %f', epoch, iter_idx+1, sum(closs)/len(closs))
            


import editdistance
# slow implementation
def test(epoch, tset='test'):
    net.eval()

    if tset=='test':
        loader = test_loader
    elif tset=='val':
        loader = val_loader
    else:
        print("not recognized set in test function")

    logger.info('Testing ' + tset + ' set at epoch %d', epoch)

    tdecs = []
    transcrs = []
    for (img, transcr) in loader:
        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            o = net(img)
        tdec = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        tdecs += [tdec]
        transcrs += list(transcr)

    tdecs = np.concatenate(tdecs)

    cer, wer = [], []
    cntc, cntw = 0, 0
    for tdec, transcr in zip(tdecs, transcrs):
        transcr = transcr.strip()
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
        dec_transcr = dec_transcr.strip()

        # calculate CER and WER
        cc = float(editdistance.eval(dec_transcr, transcr))
        ww = float(editdistance.eval(dec_transcr.split(' '), transcr.split(' ')))
        cntc += len(transcr)
        cntw +=  len(transcr.split(' '))
        cer += [cc]
        wer += [ww]

    cer = sum(cer) / cntc
    wer = sum(wer) / cntw

    logger.info('CER at epoch %d: %f', epoch, cer)
    logger.info('WER at epoch %d: %f', epoch, wer)

    net.train()


def test_both(epoch, tset='test'):
    net.eval()

    if tset=='test':
        loader = test_loader
    elif tset=='val':
        loader = val_loader
    else:
        print("not recognized set in test function")

    logger.info('Testing ' + tset + ' set at epoch %d', epoch)

    tdecs_rnn = []
    tdecs_cnn = []
    #tdecs_merge = []
    transcrs = []
    for (img, transcr) in loader:
        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            o, aux_o = net(img)

        tdec = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        tdecs_rnn += [tdec]

        tdec = aux_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        tdecs_cnn += [tdec]

        #tdec = (o + aux_o).argmax(2).permute(1, 0).cpu().numpy().squeeze()
        #tdecs_merge += [tdec]

        transcrs += list(transcr)

    cases = ['rnn', 'cnn'] #, 'merge']
    tdecs_list = [np.concatenate(tdecs_rnn), np.concatenate(tdecs_cnn)] #, np.concatenate(tdecs_merge)]

    for case, tdecs in zip(cases, tdecs_list):
        logger.info('Case: %s', case)
        cer, wer = [], []
        cntc, cntw = 0, 0
        for tdec, transcr in zip(tdecs, transcrs):
            transcr = transcr.strip()
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
            dec_transcr = dec_transcr.strip()

            # calculate CER and WER
            cc = float(editdistance.eval(dec_transcr, transcr))
            ww = float(editdistance.eval(dec_transcr.split(' '), transcr.split(' ')))
            cntc += len(transcr)
            cntw +=  len(transcr.split(' '))
            cer += [cc]
            wer += [ww]

        cer = sum(cer) / cntc
        wer = sum(wer) / cntw

        logger.info('CER at epoch %d: %f', epoch, cer)
        logger.info('WER at epoch %d: %f', epoch, wer)

    net.train()

cnt = 1
logger.info('Training:')
#test(0)
for epoch in range(1, max_epochs + 1):

    train(epoch)
    scheduler.step()

    if epoch % 10 == 0:
        if head_type=="both":
            if val_set is not None:
                test_both(epoch, 'val')
            test_both(epoch, 'test')
        else:
            if val_set is not None:
                test(epoch, 'val')
            test(epoch, 'test')

    #if epoch % 10 == 0:
    #    logger.info('Saving net after %d epochs', epoch)
    #     torch.save(net.cpu().state_dict(), 'temp.pt')
    #    net.cuda(gpu_id)


    if 'cos' in args.scheduler:
        if epoch % restart_epochs == 0:
            parameters = list(net.parameters())
            optimizer = torch.optim.AdamW(parameters, nlr, weight_decay=0.00005)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, restart_epochs)
