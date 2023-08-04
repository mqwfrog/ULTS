import torchvision.models as models
import torch
from models.mlp_head import MLPHead
from torch import nn
from config_files.epilepsy_Configs import Config as Configs

class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
            # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
            #                        bias=False) 
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=1, padding=43,
                                   bias=False) #epilepsy
            # resnet.conv1 = nn.Conv2d(9, 64, kernel_size=8, stride=1, padding=4,
            #                          bias=False) #HAR
            # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=1, padding=4,
            #                          bias=False)  # SHAR
            # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=1, padding=4,
            #                          bias=False)  # wisdm
            print(resnet)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
