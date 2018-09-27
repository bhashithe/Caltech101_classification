import torch
from torchvision import transforms
from Caltech101Data import Caltech101Data
from classifier import Classifier
from Trainer import Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam

tr = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.Resize((224,224))])
model = Classifier(102) # or you can torch.load(model_complete.mdl)
cd = Caltech101Data('image_label', tr)
optimizer = Adam(model.parameters())
loss_function = torch.nn.CrossEntropyLoss()

#'''
# if you want to load a checkpoint of a model
checkpoint = torch.load('model_checkpoint.mdl')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
#'''

loader = DataLoader(cd, batch_size=20, shuffle=True)
trainer = Trainer(loader, cd, optimizer, loss_function, model)

def main():
	trainer.train(2, loss)

if __name__ == '__main__':
	main()
