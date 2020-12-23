import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return  self.leakyrelu(self.batchnorm(self.conv(x)))

class YoloV1(nn.Module):
    def __init__(self, in_channels = 3, **kwargs):
        super(YoloV1, self).__init__()
        self.architecture = [
            # CONV layer: kernel size, out_cannels, stride, padding
            # MAX pooling: kernel size, stride
            ('conv', (7, 64, 2, 3)),
            ('maxpool', (2,2)),
            ('conv', (3, 192, 1, 1)),
            ('maxpool',(2,2)),
            ('conv', (1, 128, 1, 0)),
            ('conv', (3, 256, 1, 1)),
            ('conv', (1, 256, 1, 0)),
            ('conv', (3, 512, 1, 1)),
            ('maxpool', (2, 2)),
            ('block', ([
                ('conv', (1, 256, 1, 0)),
                ('conv', (3, 512, 1, 1))
                ],
                4
            )),
            ('conv', (1, 512, 1, 0)),
            ('conv', (3, 1024, 1, 1)),
            ('maxpool', (2, 2)),
            ('block', ([
                ('conv', (1, 512, 1, 0)),
                ('conv', (3, 1024, 1, 1))
                ],
                2
            )),
            ('conv', (3, 1024, 1, 1)),
            ('conv', (3, 1024, 2, 1)),
            ('conv', (3, 1024, 1, 1)),
            ('conv', (3, 1024, 1, 1)),
        ]
        self.in_channels = in_channels
        self.features = self._create_features()
        self.classifier = self._create_fcs(**kwargs)

    def _create_features(self):
        layers = []
        in_channels = self.in_channels
        for layer_type, layer_cfg in self.architecture:
            if layer_type == 'conv':
                k_size, out_cannels, stride, pad = layer_cfg
                layers.append(
                    CNNBlock(in_channels, out_cannels, kernel_size = k_size, stride = stride, padding = pad)
                )
                in_channels = out_cannels
            elif layer_type == 'maxpool':
                k_size, stride = layer_cfg
                layers.append(
                    nn.MaxPool2d(kernel_size=k_size, stride=stride)
                )
            elif layer_type == 'block':
                blocks, num_repeats = layer_cfg
                for n in range(num_repeats):
                    for b_layer_type, b_layer_cfg in blocks:
                        if b_layer_type == 'conv':
                            k_size, out_cannels, stride, pad = b_layer_cfg
                            layers.append(
                                CNNBlock(in_channels, out_cannels, kernel_size=k_size, stride=stride, padding=pad)
                            )
                            in_channels = out_cannels
                        else: # layer_type == 'maxpool':
                            k_size, stride = b_layer_cfg
                            layers.append(
                                nn.MaxPool2d(kernel_size=k_size, stride=stride)
                            )
        return nn.Sequential(*layers)

    def _create_fcs(self, grid_size, num_boxes, num_classes):
        S, B, C = grid_size, num_boxes, num_classes
        out = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(1024 * S * S, 4960), # TODO: add in_channel; !!! can change for fast training 4960 to 496
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4960, S * S * (C + B * 5))
        )
        return out

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)
        return x


def test(S = 7, B = 2, C = 20):
    model = YoloV1(grid_size = S, num_boxes = B, num_classes = C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)


if __name__=='__main__':
	test()