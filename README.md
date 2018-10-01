---
title: TCV Image data object detection problem
---

# Problem statement

There is a 6000 image data set which we should apply an object detection problem to identify multiple objects in one image

# Todo
- [x] Create the data abstraction using PyTorch
- [x] Create training interface
- [x] Introduce validation while training
- [x] Create the interface for saving and loading models using checkpoints
- [ ] Train the Caltech101 set
- [ ] Test the model with predictions, reruns and hyper-parametrs

# Issues

- [x] *RESOLVED* TypeError: 'builtin_function_or_method' object is not iterable
	- transforms
	- _Using PIL instead of CV2_
	- Check transforms before applying them

- [x] *RESOLVED* TypeError: batch must contain tensors, numbers, dicts or lists; found <class 'PIL.JpegImagePlugin.JpegImageFile'>
	- Probably due to the transformation again
	- _ToTensor() at the end of the transformation_

- [x] *RESOLVED* RuntimeError: size mismatch, m1: [453690 x 128], m2: [64 x 10] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:2070 
	- Due to output shape sizes
	- _Take the shape of the forward pass output shape_
