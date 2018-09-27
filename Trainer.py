import torch
from torch.optim import Adam
from classifier import Classifier
from torch.utils.data import DataLoader

class Trainer():
	def __init__(self, loader, data, optimizer, loss_function, model):
		self.model = model
		self.loader = loader
		self.data = data
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.model = model

	def train(self,epochs):
		for epoch in range(epochs):
			running_loss = 0.0
			for batch_idx, data in enumerate(self.loader,0):
				inputs, labels = data['image'], data['label']

				self.optimizer.zero_grad()

				print("labels: ", type(labels))
				outputs = self.model(inputs)
				loss = self.loss_function(outputs, labels)
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()

				if batch_idx%2000==0:
					print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/2000))
					running_loss = 0.0
		print('finished training')


