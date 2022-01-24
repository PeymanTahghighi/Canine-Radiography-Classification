#===============================================================
#===============================================================
from cgitb import reset
from copy import deepcopy
from shutil import copy
import torch
import torch.nn as nn
from torchvision import models
#===============================================================
#===============================================================

#---------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__();
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False, padding=1),
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
class Resnet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__();
        self.model = models.resnet18(pretrained= True);
        self.model.fc = nn.Linear(512, num_classes, bias=True);
    
    def forward(self, inp):
        out = self.model(inp);

        return out;
#---------------------------------------------------------------

#---------------------------------------------------------------
def test():

    sample = torch.rand((1,3,512,512));
    net = Resnet(5);
    out = net(sample);
    print(out.size());
#---------------------------------------------------------------

#---------------------------------------------------------------
if __name__ == "__main__":
    test();
#---------------------------------------------------------------