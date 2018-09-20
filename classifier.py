import torch
from torch import nn
from torch.nn import functional
import torch.optim as optim

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.conv1 = nn.Conv2d(3,6,5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(16*5*5,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self,x):
		x = self.pool(functional.relu(self.conv1(x)))
		x = self.pool(functional.relu(self.conv2(x)))
		x = x.view(-1,16*5*5)
		x = functional.relu(self.fc1(x))
		x = functional.relu(self.fc2(x))
		x = self.fc3(x)
		
		return x


class Trainer():
	classifier = Classifier()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
