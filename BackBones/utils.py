from torch import nn
from torch.nn.init import kaiming_normal_
def init_weights(model):
	for layer in model.features:
		if type(layer) in [nn.Conv2d, nn.Linear]:
			kaiming_normal_(layer.weight)
	for layer in model.classifier:
		if type(layer) in [nn.Conv2d, nn.Linear]:
			kaiming_normal_(layer.weight)
	return model