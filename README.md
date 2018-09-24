---
title: TCV Image data object detection problem
---

# Problem statement

There is a 6000 image data set which we should apply an object detection problem to identify multiple objects in one image

# Todo
[+] Create the data abstraction using PyTorch
[-] Create neural network abstraction
[-] Create training interface
[-] Train the Caltech101 set
[-] Data labeled

# Issues

[+] TypeError: 'builtin_function_or_method' object is not iterable
	- **transforms**
	- Using PIL instead of CV2
	- Check transforms before applying them
