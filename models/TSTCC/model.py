from torch import nn
from config_files.epilepsy_Configs import Config as Configs 
import torch
class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
# Original-HAR
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        print(f'x_in.shape:{x_in.shape}') # har: [128, 9, 128] & shar: [128,3,151] $ wisdm: [128,3,256]
        x = self.conv_block1(x_in)
        print(x.shape) #[128, 32, 65] & [128, 32, 77] & [128, 32, 129]
        x = self.conv_block2(x)
        print(x.shape) #[128, 64, 34] & [128, 64, 40]) & [128, 64, 66]
        x = self.conv_block3(x)
        print(x.shape) #[128, 128, 18] & [128, 128, 21] & [128, 128, 34] 

        x_flat = x.reshape(x.shape[0], -1)
        print(f'x_flat.shape:{x_flat.shape}')
        logits = self.logits(x_flat)
        return logits, x
    #   logits [128-batch_size, 2-num_classes]
    #   x      [128-batch_size, 128-final_out_channels, 24-features_len]

if __name__ == '__main__':
    finalconfigs = Configs()
    m = base_Model(finalconfigs)
    x = torch.randn(32,1,178)
    y = m(x)[1]
    print(y.shape) #[128, 128, 24]