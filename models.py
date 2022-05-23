import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class CNN(nn.Module):
    def __init__(self, cnn_cfg, flattening='maxpool'):
        super(CNN, self).__init__()

        self.k = 1
        self.flattening = flattening

        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, [2, 2], 3), nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x,))
                    in_channels = x
                    cnt += 1

    def forward(self, x, reduce='max'):

        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

        if self.flattening=='maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
        elif self.flattening=='concat':
            y = y.view(y.size(0), -1, 1, y.size(3))

        return y

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


class CTCtopC(nn.Module):
    def __init__(self, input_size, head_cfg, nclasses):
        super(CTCtopC, self).__init__()

        hidden_size, num_layers = head_cfg

        self.temporal_i = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=(1,5), stride=(1,1), padding=(0,2)),
            nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.25),
        )

        list = [
            nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
                nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.25),
            ) for _ in range(num_layers)
        ]

        self.temporal_m = nn.ModuleList(list)

        self.temporal_o = nn.Conv2d(hidden_size, nclasses, kernel_size=(1, 5), stride=1, padding=(0, 2))


    def forward(self, x):

        y = self.temporal_i(x)

        for f in self.temporal_m:
            y = f(y)

        y = self.temporal_o(y)
        y = y.permute(2, 3, 0, 1)[0]
        return y

class CTCtopR(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses):
        super(CTCtopR, self).__init__()

        hidden, num_layers = rnn_cfg

        self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        self.fnl = nn.Sequential(nn.Dropout(.2), nn.Linear(2 * hidden, nclasses))

    def forward(self, x):

        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]
        y = self.fnl(y)

        return y

class CTCtopB(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses):
        super(CTCtopB, self).__init__()

        hidden, num_layers = rnn_cfg

        self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        self.fnl = nn.Sequential(nn.Dropout(.2), nn.Linear(2 * hidden, nclasses))

        self.att = nn.Sequential(nn.Dropout(.5), nn.Linear(2 * hidden, nclasses), nn.Sigmoid())

        self.cnn = nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):

        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]
        #a = self.att(y)
        #y = (a * self.fnl(y) + (1 - a) * self.cnn(x))
        y = self.fnl(y)

        if self.training:
            return y, self.cnn(x).permute(2, 3, 0, 1)[0]
        else:
            return y, self.cnn(x).permute(2, 3, 0, 1)[0]


class STN(nn.Module):

    def __init__(self):
        super(STN, self).__init__()

        cfg = [(2, 16), 'M', (2, 32), 'M', (2, 64)]
        self.cnn = CNN(cfg)

        self.aff_last = nn.Linear(cfg[-1][-1], 6)
        self.aff_last.weight.data *= 0.1
        self.aff_last.bias.data *= 0.1

        self.ref_aff = torch.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


    def forward(self, x):
        xd = x
        y = self.cnn(x, reduce=False)

        y = F.adaptive_avg_pool2d(y, [1, 1])
        y_aff = self.aff_last(y.view(x.size(0), -1))

        aff = self.ref_aff.to(x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
        #aff[:, :, :2] += y_aff.view(-1, 2, 2)
        aff += y_aff.view(-1, 2, 3)

        grid = F.affine_grid(aff, x.size())
        xd = F.grid_sample(xd, grid, padding_mode='border')

        return xd

class HTRNet(nn.Module):
    def __init__(self, cnn_cfg, head_cfg, nclasses, head='cnn', stn=False, flattening='maxpool'):
        super(HTRNet, self).__init__()

        if stn: 
            self.stn = STN()
        else:
            self.stn = None

        self.features = CNN(cnn_cfg, flattening=flattening)

        if flattening=='maxpool':
            hidden = cnn_cfg[-1][-1]
        elif flattening=='concat':
            hidden = 2 * 8 * cnn_cfg[-1][-1]
        else:
            print('problem!')

        if head=='cnn':
            self.top = CTCtopC(hidden, head_cfg, nclasses)
        elif head=='rnn':
            self.top = CTCtopR(hidden, head_cfg, nclasses)
        elif head=='both':
            self.top = CTCtopB(hidden, head_cfg, nclasses)

    def forward(self, x):

        if self.stn is not None:
            x = self.stn(x)

        y = self.features(x)
        y = self.top(y)

        return y


def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()