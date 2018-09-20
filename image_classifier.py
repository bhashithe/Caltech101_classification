"""
Purpose:This is a image classification test 
Data : 	Using CIFAR data from ![](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
	Check info at ![](https://www.cs.toronto.edu/~kriz/cifar.html)
Author: @bhashithe_a
"""

import numpy as np
import joblib
import torch


def load_data(inpath):
	"""
	@inpath: string path for the joblib files to be loaded
	returns => the loaded pickle file
	"""
	return joblib.load(inpath)

def main():
	data = load_data('images/data_batch_1')

if __name__ == '__main__':
	main()
