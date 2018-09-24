import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class Caltech101Data(Dataset):
	""" Caltech101 dataset """

	def __init__(self, inpath, transform=None):
		self.transform = transform
		self.path = inpath
		self.image_list = os.listdir(self.path)

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		print(self.image_list[idx])
		image_path = os.path.join(self.path,self.image_list[idx])
		image = Image.open(image_path)
		label = self.image_list[idx].split('_image')[0]

		if self.transform:
			self.transform(image)
		sample = {'image': image, 'label': label}
		
		return sample
