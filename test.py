import torch
from torchvision import transforms
from Caltech101Data import Caltech101Data
from classifier import Classifier
from Trainer import Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam

tr = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.Resize((224,224))])

cd = Caltech101Data('image_label', tr)

model = Classifier(102)
adam = Adam(model.parameters())
loss_function = torch.nn.CrossEntropyLoss()

loader = DataLoader(cd, batch_size=20, shuffle=True)
trainer = Trainer(loader, cd, adam, loss_function, model)

def main():
	trainer.train(2)

if __name__ == '__main__':
	main()
