import numpy as np
import pandas as pd

def relu(inputs):
    """
    Performs the ReLU optimization on an array
    
    Parameters:
    ___________
    inputs = The array in which ReLU should be done.
    
    Returns:
    ________
    Returns the array of same length in which ReLU is applied
    
    """
    
    return(np.maximum(inputs,0))

def relu_derivative(Z):
    #because Z is a Vector
    #https://math.stackexchange.com/a/368445
    #https://math.stackexchange.com/questions/368432/derivative-of-max-function
    
    temp = np.where(Z<0, Z, 1) #replace a greater than 0 with 1
    final = np.where(temp>0, temp, 0) #replace temp smaller than 0 with 0
    
    return final;

def sigmoid(inputs):
    op = 1/(1+np.exp(-inputs))
    return op

def sigmoid_derivative(Z):
    s = sigmoid(Z)
    calc = s*(1-s)
    return calc

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_derivative(Z):
    

def initialize_parameters(size, debug=0):
    """
    Creates the parameters for given size. Initiates using He Initialization.
    
    Parameters:
    ___________
    size: int array
    
        The number of neurons in each layer.
        Example:
        [5,4,1] creates layers of 5 input layer, 4 hidden and 1 output
    
    debug: int (optional)
        
        If set to 1, np.random.seed(1) is used while generating the variables
        
    Returns:
    ________
    parameters: Python Dictionary
        A python dictonary containing initialized W and b
    """
    
    if (debug == 1):
        np.random.seed(1) #for debugging purpose
   
    parameters = {}
    no_layers = len(size) #as size contains size of each layers
    
    for i in range(1, no_layers):
        parameters['W' + str(i)] = np.random.randn(size[i], size[i-1])  * np.sqrt(2./(size[i-1])) #He initialization
        parameters['b' + str(i)] = np.zeros((size[i],1))
        
        assert(parameters['W' + str(i)].shape == (size[i], size[i-1]))
        assert(parameters['b' + str(i)].shape == (size[i], 1))
        
    return parameters

def cost_function(AL, y, lambd=0, parameters={}):
    """
    Calculates logistic regression cost
    
    Parameters:
    ___________
    y: numpy array with same dimension as AL
        The actual output in which AL should be compared
    AL: numpy array
        The final layer calculated by forward propagation.
    lambd: int (optional)
        If set to a number, regularization is done
    parameters: dictionary (optional)
        The dictionary of parameters W and b. Sent when regularization is done.
    
    Returns:
    ________
    cost: float
        The Cost calculated. Single number.
    """
    
    m = y.shape[1]
    
    cost = (1./m) * (-np.dot(y,np.log(AL).T) - np.dot(1-y, np.log(1-AL).T))
    cost = np.sum(cost) #might be sth wrong. replace the cost with that for one hot
    cost = np.squeeze(cost)        
    reg = 0;
    
    if (lambd != 0):
        lengthP = len(parameters) // 2
    
        for i in range(1,lengthP+1):
            reg = reg + np.sum(np.square(parameters['W' + str(i)]))
        
        reg = reg * (lambd/(2*m))
        
    cost = cost + reg
    cost = np.squeeze(cost)
    
    assert(cost.shape == ()) #The value returned must be a single number
    return cost

def forward_propagation(X, parameters):
    """"
    Computes forward propagation
    
    Parameters:
    ___________
    X: numpy array
        Array of inputs in the specified format.
    
    parameters: Dictonary
        Values of W and b
    
    Returns:
    ________
    cache: Dictionary
        Values of Z (Product) and Activation 
    AL: Numpy array
        The final layer predicted
    """
    
    amt = len(parameters)//2 #because we just need half as half is w and half is b
    cache = {}
    A_prev = X
    
    for i in range(1, amt):
        cache['Z' + str(i)] = np.dot(parameters['W' + str(i)], A_prev) + parameters['b' + str(i)]
        cache['A' + str(i)] = relu(cache['Z' + str(i)])
        A_prev = cache['A' + str(i)]
    
    #outside loop because last output should be probablity. Therefore sigmoid insted of r
    cache['Z' + str(amt)] = np.dot(parameters['W' + str(amt)], A_prev) + parameters['b' + str(amt)]
    cache['A' + str(amt)] = sigmoid(cache['Z' + str(amt)])
    
    AL = cache['A' + str(amt)]
    
    return cache,AL

def back_propagation(X, Y, cache,parameters,lambd=0):
    """
    Performs Back Propagation
    
    Parameters:
    ___________
    X: numpy array of inputs
    Y: numpy array of real outputs to train
    cache: numpy array containing Z and A from forward propagation
    parameters: parameters of w and b
    lambd: (optional) If regularize
    
    Returns:
    ________
    grads: Dictionary containing dA, dZ, dW, db
    """
    
    grads = {}
    length = len(cache) // 2 #half elements are A, half is Z as above
    m = Y.shape[1]
    
    dAL =  -(np.divide(Y, cache['A' + str(length)]) - np.divide(1 - Y, 1 - cache['A' + str(length)]))
    
    grads['dA' + str(length)] = dAL
    grads['dZ' + str(length)] = dAL * sigmoid_derivative(cache['Z' + str(length)])
    grads['dW' + str(length)] = (np.dot(grads['dZ' + str(length)], cache['A' + str(length-1)].T)/m) + ((lambd/m) * parameters['W'+ str(length)]) #chain rule from last
    grads['db' + str(length)] = np.sum(grads['dZ' + str(length)], axis=1, keepdims=True)/m
    
    dA_prev = np.dot(parameters['W' + str(length)].T, grads['dZ' + str(length)])
    
    cache['A0'] = X #because for loop automaticall comes front from back and A0 is not directly understood
    
    assert(grads['dA' + str(length)].shape == cache['A' + str(length)].shape)
    assert(grads['dZ' + str(length)].shape == cache['Z' + str(length)].shape)
    assert(grads['dW' + str(length)].shape == parameters['W' + str(length)].shape)
    assert(grads['db' + str(length)].shape == parameters['b' + str(length)].shape) 
    
    for i in reversed(range(1,length)):
        m = cache['A' + str(i)].shape[1]
        A_prev = cache['A' + str(i-1)]
        
        grads['dZ' + str(i)] = dA_prev * relu_derivative(cache['Z' + str(i)])
        grads['dW' + str(i)] = (np.dot(grads['dZ' + str(i)], A_prev.T)/m) + ((lambd/m) * parameters['W'+ str(i)])
        grads['db' + str(i)] = np.sum(grads['dZ' + str(i)], axis=1, keepdims=True)/m
        grads['dA' + str(i)] = dA_prev
        
        assert(grads['dZ' + str(i)].shape == cache['Z' + str(i)].shape)
        assert(grads['dW' + str(i)].shape == parameters['W' + str(i)].shape)
        assert(grads['db' + str(i)].shape == parameters['b' + str(i)].shape) 
        assert(grads['dA' + str(i)].shape == cache['A' + str(i)].shape) 
        
        dA_prev = np.dot(parameters['W' + str(i)].T, grads['dZ' + str(i)])
    
    
    return grads;