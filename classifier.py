import torch
from torch import nn
from torch.nn import functional

class ConvUnit(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ConvUnit, self).__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
		self.norm = nn.BatchNorm2d(num_features=out_channels)
		self.acti = nn.ReLU()
	
	def forward(self, input):
		output = self.conv(input)
		output = self.norm(output)
		output = self.acti(output)

		return output

class Classifier(nn.Module):
	def __init__(self, num_classes):
		super(Classifier, self).__init__()
		self.unit0 = ConvUnit(3, 32)
		self.unit1 = ConvUnit(32, 32)
		self.pool0 = nn.MaxPool2d(2, stride=1)
		self.unit2 = ConvUnit(32, 64)
		self.unit3 = ConvUnit(64, 64)
		self.pool2 = nn.AvgPool2d(3, stride=1)
		self.network = nn.Sequential(self.unit0, self.unit1, self.pool0, self.unit2, self.unit3, self.pool1)

		self.fc = nn.Linear(in_features=64, num_classes=num_classes)

	def forward(self, input):
		output = self.network(input)
		output = self.fc(output.view(-1, 128))

		return output
