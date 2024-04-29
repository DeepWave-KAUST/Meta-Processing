import torch
import torch.nn as nn
import torch.nn.functional as F

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.relu(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.relu(x)
        x = self.conv5(x)

        return y + x * self.gamma

def Functional_NAFBlock(inp, bn_weights0, bn_biases0, bn_weights1, bn_biases1, weights0, biases0, 
            weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4, weights5, biases5, beta, gamma):

        # self.norm1
        x = F.batch_norm(inp, running_mean=None, running_var=None, weight=bn_weights0, bias=bn_biases0, training=True)
        # self.conv1
        x = F.conv2d(x, weights0, biases0)
        # self.conv2
        x = F.conv2d(x, weights1, biases1, padding=1, groups=x.size()[1])
        # self.sg
        x1 = F.leaky_relu(x, inplace=True, negative_slope=0.1)
        # self.sca
        x = F.adaptive_avg_pool2d(x1,1)
        x = F.conv2d(x, weights2, biases2)
        x = x * x1
        # self.conv3
        x = F.conv2d(x, weights3, biases3)

        inp = inp + x * beta
        # self.norm2
        x = F.batch_norm(inp, running_mean=None, running_var=None, weight=bn_weights1, bias=bn_biases1, training=True)
        # self.conv4
        x = F.conv2d(x, weights4, biases4)
        # self.sg
        x = F.leaky_relu(x, inplace=True, negative_slope=0.1)
        # self.conv5
        x = F.conv2d(x, weights5, biases5)
        # 
        x = inp + x * gamma
        return x

class METANAFNet(nn.Module):

    def __init__(self, in_channels=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=in_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.middle_blk_num = middle_blk_num
        self.enc_blk_nums = enc_blk_nums
        self.dec_blk_nums = dec_blk_nums

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def functional_forward(self, inp, params):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        
        x = F.conv2d(inp, params['intro.weight'], params['intro.bias'], padding=1)

        encs = []

        id = 0
        for num in self.enc_blk_nums:
            for i in range(num):
                x = Functional_NAFBlock(x,  
                    params['encoders.{}.{}.norm1.weight'.format(id,i)], params['encoders.{}.{}.norm1.bias'.format(id,i)],
                    params['encoders.{}.{}.norm2.weight'.format(id,i)], params['encoders.{}.{}.norm2.bias'.format(id,i)],
                    params['encoders.{}.{}.conv1.weight'.format(id,i)], params['encoders.{}.{}.conv1.bias'.format(id,i)],
                    params['encoders.{}.{}.conv2.weight'.format(id,i)], params['encoders.{}.{}.conv2.bias'.format(id,i)],
                    params['encoders.{}.{}.sca.1.weight'.format(id,i)], params['encoders.{}.{}.sca.1.bias'.format(id,i)],
                    params['encoders.{}.{}.conv3.weight'.format(id,i)], params['encoders.{}.{}.conv3.bias'.format(id,i)],
                    params['encoders.{}.{}.conv4.weight'.format(id,i)], params['encoders.{}.{}.conv4.bias'.format(id,i)],
                    params['encoders.{}.{}.conv5.weight'.format(id,i)], params['encoders.{}.{}.conv5.bias'.format(id,i)],
                    params['encoders.{}.{}.beta'.format(id,i)], params['encoders.{}.{}.gamma'.format(id,i)])
            encs.append(x)
            x = F.conv2d(x, params['downs.{}.weight'.format(id)], params['downs.{}.bias'.format(id)], stride=2)
            id = id + 1

        for num in range(self.middle_blk_num):
            x = Functional_NAFBlock(x, 
                    params['middle_blks.{}.norm1.weight'.format(num)], params['middle_blks.{}.norm1.bias'.format(num)],
                    params['middle_blks.{}.norm2.weight'.format(num)], params['middle_blks.{}.norm2.bias'.format(num)],
                    params['middle_blks.{}.conv1.weight'.format(num)], params['middle_blks.{}.conv1.bias'.format(num)],
                    params['middle_blks.{}.conv2.weight'.format(num)], params['middle_blks.{}.conv2.bias'.format(num)],
                    params['middle_blks.{}.sca.1.weight'.format(num)], params['middle_blks.{}.sca.1.bias'.format(num)],
                    params['middle_blks.{}.conv3.weight'.format(num)], params['middle_blks.{}.conv3.bias'.format(num)],
                    params['middle_blks.{}.conv4.weight'.format(num)], params['middle_blks.{}.conv4.bias'.format(num)],
                    params['middle_blks.{}.conv5.weight'.format(num)], params['middle_blks.{}.conv5.bias'.format(num)],
                    params['middle_blks.{}.beta'.format(num)], params['middle_blks.{}.gamma'.format(num)])

        id = 0
        for num in self.dec_blk_nums:
            x = F.conv2d(x, params['ups.{}.0.weight'.format(id)])
            x = F.pixel_shuffle(x,2)
            x = x + encs[3-id]
            for i in range(num):
                x = Functional_NAFBlock(x,  
                    params['decoders.{}.{}.norm1.weight'.format(id,i)], params['decoders.{}.{}.norm1.bias'.format(id,i)],
                    params['decoders.{}.{}.norm2.weight'.format(id,i)], params['decoders.{}.{}.norm2.bias'.format(id,i)],
                    params['decoders.{}.{}.conv1.weight'.format(id,i)], params['decoders.{}.{}.conv1.bias'.format(id,i)],
                    params['decoders.{}.{}.conv2.weight'.format(id,i)], params['decoders.{}.{}.conv2.bias'.format(id,i)],
                    params['decoders.{}.{}.sca.1.weight'.format(id,i)], params['decoders.{}.{}.sca.1.bias'.format(id,i)],
                    params['decoders.{}.{}.conv3.weight'.format(id,i)], params['decoders.{}.{}.conv3.bias'.format(id,i)],
                    params['decoders.{}.{}.conv4.weight'.format(id,i)], params['decoders.{}.{}.conv4.bias'.format(id,i)],
                    params['decoders.{}.{}.conv5.weight'.format(id,i)], params['decoders.{}.{}.conv5.bias'.format(id,i)],
                    params['decoders.{}.{}.beta'.format(id,i)], params['decoders.{}.{}.gamma'.format(id,i)])
            id = id + 1

        x = F.conv2d(x, params['ending.weight'], params['ending.bias'], padding=1) + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x