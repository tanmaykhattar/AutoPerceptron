import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy import stats

# Settings

def func(x):
    return np.sin(x * np.pi) # Modify this line to add your function

layer_max = 10
neuron_max = 200 # Minimum of 10 neurons per layer
num_epochs = 100
activation_function = 'relu' # More options here: keras.io/activations

# Generate Training and Test Data

x_train = np.linspace(-1,1,101) # Odd numbers 
x_test = np.linspace(-0.99,0.99,100) # Even numbers
y_train = list(map(func, x_train))
y_test = list(map(func, x_test))

# Format Data for NN

x_train = [[elem] for elem in x_train]
x_test = [[elem] for elem in x_test]
y_train = [[elem] for elem in y_train]
y_test = [[elem] for elem in y_test]


def basic_model(num_layers = 3, num_neurons = 256):

    # Create Model

    model = Sequential()

    # Add Hidden Layers

    for _ in range(num_layers):
        model.add(Dense(num_neurons, input_dim = 1, kernel_initializer='normal', activation = activation_function))
    model.add(Dense(1, kernel_initializer='normal'))
    
    # Compile
    
    model.compile('adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    return model

def rsq(x,y): # Find rsq

    # Flatten List
    
    x = [elem[0] for elem in x]
    y = [elem[0] for elem in y]

    _, _, r_value, _, _ = stats.linregress(x, y)
    return r_value**2

rsq_matrix = [None for _ in range(neuron_max // 10 * layer_max)] # Initialize empty list for rsq entries
counter = 0

# Optimize Neural Network Architechture by cycling through configurations of layers/neurons

for neuron_indx in range(neuron_max // 10): # We cycle through the neurons in increments of 10 since there are likely many more neurons than layers
    for layer_indx in range(layer_max):

        # Convert indx to number of layers/neurons 

        layers = layer_indx + 1
        neurons = (neuron_indx + 1) * 10

        # Calculate predictions of model
        
        model = basic_model(layers, neurons)
        model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = num_epochs, verbose = 0)
        y_model = model.predict(x_test)
        
        # Compute rsq value and put in array
        
        rsq_matrix[counter] = [layers, neurons, rsq(y_model, y_test)]

        # Printing Progress
        
        counter += 1
        print('step %d of %d'%(counter,(neuron_max // 10 * layer_max))) 


layers_opt, neurons_opt,  _ = max(rsq_matrix, key = lambda x: x[2]) # Find optimal layers and neurons from array

#Generate and Save Model

opt_model = basic_model(layers_opt, neurons_opt)
opt_model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, verbose = 0)
opt_model.save('my_model.h5')

#Plot Information

plt.plot(x_test, model.predict(x_test), 'rx', x_test, y_test, 'kx')
plt.legend(['Target', 'Model Prediction'])
plt.title('Optimal Model Configuration (rsq = %.3f): %d Layers, %d Neurons'%(round(rsq(model.predict(x_test),y_test),3),layers,neurons))
plt.show()












