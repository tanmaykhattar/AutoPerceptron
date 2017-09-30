# AutoPerceptron

This script automates the design of the architecture of a multilayer 
perceptron for one-dimentional function appoximation. Given a function which 
takes a domain [-1, 1], the script finds the optimal number of layers and neurons 
per layer of the percepteron. It returns the trained optimal model and 
save it to the current directory as a HDF5 file 'my_model.h5'. It also creates 
a plot of the model performance showing how well it can generalize beyond its 
training data. 

The script assumes that numpy, keras, matplotlib, and scipy and their 
dependencies have been installed correctly
