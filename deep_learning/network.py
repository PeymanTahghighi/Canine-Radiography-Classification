#===============================================================
#===============================================================
from copy import deepcopy
from shutil import copy
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
#===============================================================
#===============================================================

#---------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__();
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.net(x);
#---------------------------------------------------------------

#---------------------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, type) -> None:
        super().__init__();

        if type == 'convtrans':
            self.net = nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=2, padding=1);
        else:
            self.net = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=1)
            )
        
        self.conv_after = nn.Sequential(
            ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1),
            ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1))
    
    def forward(self, x):
        x = self.net(x);
        x = self.conv_after(x);
        return x;
#---------------------------------------------------------------

#---------------------------------------------------------------
class Upblock(nn.Module):
    def __init__(self, in_features, out_features, concat_features = None) -> None:
        super().__init__();
        if concat_features == None:
            concat_features = out_features*2;

        self.upsample = Upsample(in_features, out_features, 4, 'convtrans');
        self.conv1 = ConvBlock(in_channels=concat_features, out_channels=out_features, kernel_size=3, stride=1);
        self.conv2 = ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1);

    def forward(self, x1, x2):

        x1 = self.upsample(x1);
        ct = torch.cat([x1,x2], dim=1);
        ct = self.conv1(ct);
        out = self.conv2(ct);
        return out;
#---------------------------------------------------------------

#---------------------------------------------------------------
class UpblockAttentionGate(nn.Module):
    def __init__(self, lower_features, upper_features, int_features = None, concat_features = None) -> None:
        super().__init__();
        if int_features == None:
            int_features = upper_features;
        
        if concat_features is None:
            concat_features = upper_features*2;

        self.upsample = Upsample(lower_features, int_features, 4, 'convtrans');
        self.conv1 = nn.Conv2d(lower_features, int_features, 1, 1, 0);
        self.conv2 = nn.Conv2d(upper_features, int_features, 1, 2, 0);
        self.conv3 = nn.Conv2d(int_features, 1, 1, 1, 0);
        self.refinement = nn.Sequential(
            ConvBlock(concat_features, int_features, 3, 1),
            ConvBlock(int_features, upper_features, 3, 1),
        )
    
    def forward(self, lower, upper):
        lower_feat = self.conv1(lower);
        upper_feat = self.conv2(upper);
        attention = F.interpolate(torch.sigmoid(self.conv3(F.relu(lower_feat+upper_feat))), (upper.shape[2], upper.shape[3]));
        attenion_out = upper * attention;

        lower_up = self.upsample(lower);
        out = self.refinement(torch.cat([lower_up, attenion_out], dim=1));
        return out;
#---------------------------------------------------------------

#---------------------------------------------------------------
class Unet(nn.Module):
    def __init__(self, num_classes = None) -> None:
        super().__init__();
        resnet = models.resnet50(pretrained= False);
        ckpt = torch.load('resnet50.pth');
        resnet.load_state_dict(ckpt);
        self.input_blocks = nn.Sequential(*list(resnet.children()))[:3];
        self.input_pool = list(resnet.children())[3];
        resnet_down_blocks = [];
        for btlnck in list(resnet.children()):
            if isinstance(btlnck, nn.Sequential):
                resnet_down_blocks.append(btlnck);
        
        self.down_blocks = nn.Sequential(*resnet_down_blocks);

        self.bottle_neck = nn.Sequential(
            ConvBlock(2048, 2048, 3, 1),
            ConvBlock(2048, 2048, 3, 1)
        );


        self.up_1 = Upblock(2048,1024);
        self.up_2 = Upblock(1024,512);
        self.up_3 = Upblock(512,256);
        self.up_4 = Upblock(256, 128, 128+64)
        self.up_5 = Upblock(128, 64, 64+3);
        
        if num_classes != None:
            self.final = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0);
        

        self.__initial_weights = deepcopy(self.state_dict());
        
    def forward(self, inp):
        d_1 = self.input_blocks(inp);

        d_1_pool = self.input_pool(d_1);
        d_2 = self.down_blocks[0](d_1_pool);
        d_3 = self.down_blocks[1](d_2);
        d_4 = self.down_blocks[2](d_3);
        d_5 = self.down_blocks[3](d_4);

        d_5 = self.bottle_neck(d_5);

        u_1 = self.up_1(d_5, d_4);
        u_2 = self.up_2(u_1, d_3);
        u_3 = self.up_3(u_2, d_2);
        u_4 = self.up_4(u_3, d_1);
        u_5 = self.up_5(u_4, inp);

        out = self.final(u_5);

        return out;
    

    def reset_weights(self):
        self.load_state_dict(self.__initial_weights);
#---------------------------------------------------------------

class CrossAttention(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__();
        self.channels = channel;
        self.ca = nn.MultiheadAttention(channel, 4, batch_first=True);
        self.ln = nn.LayerNorm([self.channels]);
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channel]),
            nn.Linear(channel, channel),
            nn.GELU(),
            nn.Linear(channel, channel)
        )
    
    def get_pos_embedding(self, tokens, channels):
        pe = torch.zeros((1,tokens, channels) , device= config.DEVICE, requires_grad=False);
        inv_freq_even = 1.0/((10000)**(torch.arange(0,channels,2) / channels));
        inv_freq_odd = 1.0/((10000)**(torch.arange(1,channels,2) / channels));
        pe[:,:,0::2] = torch.sin(torch.arange(0,tokens).unsqueeze(dim=1) * inv_freq_even.unsqueeze(dim=0));
        pe[:,:,1::2] = torch.cos(torch.arange(0,tokens).unsqueeze(dim=1) * inv_freq_odd.unsqueeze(dim=0));
        return pe;

    def forward(self, x1, x2):
        B,C,W,H,D = x1.shape;
        
        x1 = x1.view(B, C, W*H*D).swapaxes(1,2);
        x2 = x2.view(B, C, W*H*D).swapaxes(1,2);

        x1 = self.pos_embedding(W*H*D, C) + x1;
        x2 = self.pos_embedding(W*H*D, C) + x2;

        x1_ln = self.ln(x1);
        x2_ln = self.ln(x2);

        attntion_value, _ = self.ca(x1_ln, x1_ln, x2_ln);

        attntion_value = self.ff_self(attntion_value) + attntion_value;
        return attntion_value.swapaxes(2,1).view(B,C,W,H,D);

#---------------------------------------------------------------
class AttenUnet(nn.Module):
    def __init__(self, num_classes = None) -> None:
        super().__init__();
        resnet = models.resnet50(pretrained= False);
        ckpt = torch.load('resnet50.pth');
        resnet.load_state_dict(ckpt);
        self.input_blocks = nn.Sequential(
            ConvBlock(3,64,3,1),
            ConvBlock(64,64,3,1)
        );
        self.input_pool = list(resnet.children())[3];
        resnet_down_blocks = [];
        for btlnck in list(resnet.children()):
            if isinstance(btlnck, nn.Sequential):
                resnet_down_blocks.append(btlnck);
        
        self.down_blocks = nn.Sequential(*resnet_down_blocks);

        self.bottle_neck = nn.Sequential(
            ConvBlock(2048, 2048, 3, 1),
            ConvBlock(2048, 2048, 3, 1)
        );

        self.up_1 = UpblockAttentionGate(2048,1024);
        self.up_2 = UpblockAttentionGate(1024,512);
        self.up_3 = UpblockAttentionGate(512,256);
        self.up_4 = UpblockAttentionGate(256, 64, int_features=128, concat_features=128+64)
        
        if num_classes != None:
            self.final = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0);
        

        self.__initial_weights = deepcopy(self.state_dict());
        
    def forward(self, inp):
        d_1 = self.input_blocks(inp);

        d_1_pool = self.input_pool(d_1);
        d_2 = self.down_blocks[0](d_1_pool);
        d_3 = self.down_blocks[1](d_2);
        d_4 = self.down_blocks[2](d_3);
        d_5 = self.down_blocks[3](d_4);

        d_5 = self.bottle_neck(d_5);


        u_1 = self.up_1(d_5, d_4);
        u_2 = self.up_2(u_1, d_3);
        u_3 = self.up_3(u_2, d_2);
        u_4 = self.up_4(u_3, d_1);

        out = self.final(u_4);

        return out;
    

    def reset_weights(self):
        self.load_state_dict(self.__initial_weights);
#---------------------------------------------------------------

#---------------------------------------------------------------
def test():

    sample = torch.rand((1,3,512,512));
    net = AttenUnet(5);
    out = net(sample);
    print(out.size());
#---------------------------------------------------------------

#---------------------------------------------------------------
if __name__ == "__main__":
    test();
#---------------------------------------------------------------