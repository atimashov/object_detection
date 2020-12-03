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

class Darknet20(nn.Module):
	def __init__(self, in_channels = 3, num_classes = 1000, **kwargs):
		super(Darknet20, self).__init__()
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
			# ('conv', (3, 1024, 1, 1)),
			# ('conv', (3, 1024, 1, 1)),
		]
		self.in_channels = in_channels
		self.features = self._create_features()
		self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
		self.classifier = nn.Sequential(
			nn.Dropout(p = 0.5, inplace=False),
			nn.Linear(1024, num_classes),
		)

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

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, start_dim = 1) #x.view(x.size(0), x.size(1))  # resize for fc layers
		x = self.classifier(x)
		return x

def test():
    model = Darknet20()
    x = torch.randn((27, 3, 224, 224))
    print(model(x).shape)

if __name__=='__main__':
	test()