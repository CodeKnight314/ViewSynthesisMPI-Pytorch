import torch 
import torch.nn as nn

class ViewSynthNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, k1, c1, k2, c2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, c1, kernel_size=k1, stride=1, padding=padding),
                nn.Conv2d(c1, c2, kernel_size=k2, stride=1, padding=padding)
            )

        def down_block(in_c, k1, c1, k2, c2, padding=1):
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                conv_block(in_c, k1, c1, k2, c2, padding)
            )

        def up_block(in_c, k1, c1, k2, c2, padding=1):
            return nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                conv_block(in_c, k1, c1, k2, c2, padding)
            )

        self.encoder = nn.ModuleList([
            conv_block(3, 7, 32, 7, 32, padding=3),
            down_block(32, 5, 64, 5, 64, padding=2),
            down_block(64, 3, 128, 3, 128),
            down_block(128, 3, 256, 3, 256),
            down_block(256, 3, 512, 3, 512),
            down_block(512, 3, 512, 3, 512),
            down_block(512, 3, 512, 3, 512),
            down_block(512, 3, 512, 3, 512)
        ])

        self.decoder = nn.ModuleList([
            up_block(512, 3, 512, 3, 512),
            up_block(1024, 3, 512, 3, 512),
            up_block(1024, 3, 512, 3, 512),
            up_block(1024, 3, 512, 3, 512),
            up_block(768, 3, 128, 3, 128),
            up_block(256, 3, 64, 3, 64),
            up_block(128, 3, 64, 3, 64)
        ])

        self.conv_fifteen = conv_block(96, 3, 64, 3, 64)
        self.conv_final = nn.Conv2d(64, 34, kernel_size=3, stride=1)

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1][1:]

        for i, layer in enumerate(self.decoder):
            x = layer(x)
            x = torch.cat([x, skip_connections[i]], dim=1)

        output = self.conv_final(self.conv_fifteen(x))
        output = torch.sigmoid(output)
        return output
    
def get_ViewSynthNet(): 
    return ViewSynthNet().to("cuda" if torch.cuda.is_available() else "cpu")
    