Description: This script optimizes a multilayer perceptron to efficiently fit any 
one-dimensional function which takes the domain [-1, 1]. In addition to the 
fitting function, the user must provide the maximum number of layers, the maximum 
number of neurons per layer, the number of training epochs, and the activation 
function. The script returns a plot of the model performance against generated 
test data and saves the model to the current directory as a HDF5 file 
'my_model.h5'. To load the model, use 

>>> model = load_model('my_model.h5')

The script assumes that numpy, keras, matplotlib, and scipy and their 
dependencies have been installed correctly
