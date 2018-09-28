import torch
from torch.optim import Adam
from classifier import Classifier
from torch.utils.data import DataLoader

class Trainer():
	def __init__(self, loader, data, optimizer, loss_function, model, device):
		self.model = model
		self.loader = loader
		self.data = data
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.device = device
		self.model = model.to(self.device)

	def train(self,epochs, loss=None):
		for epoch in range(epochs):
			running_loss = 0.0
			for batch_idx, data in enumerate(self.loader,0):
				print(self.device)
				inputs, labels = data['image'], data['label']
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)
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

				# save the model state in each beatch so that it is possible to resume training later
				torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss}, 'model_checkpoint.mdl')
		print('finished training')
		# we can save the whole model
		torch.save(self.model, 'model_complete.mdl')


