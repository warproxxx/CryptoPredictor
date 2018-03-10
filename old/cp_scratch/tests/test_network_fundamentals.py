import cp_scratch.network_fundamentals as ml
import numpy as np
from cp_scratch.gradient_convertor import *

def test_relu():
    assert(ml.relu(1) == 1)
    assert(ml.relu(-1) == 0)
    assert(ml.relu(5) == 5)
    assert all(ml.relu([1,-1,0,5,-5]) == [1,0,0,5,0])
	
	
def test_initialize_parameters():
    response = ml.initialize_parameters([1,2,3])
    
    assert(response['W1'].shape == (2,1))
    assert(response['b1'].shape == (2,1))
    assert(response['W2'].shape == (3,2)) #as we need 3 rows and 2 columns
    assert(response['b2'].shape == (3,1))
    assert(len(response) == 4) #to check size
    
    response2 =  ml.initialize_parameters([5,4,3,1])
    
    assert(len(response2) == 6)
    assert(response2['W1'].shape == (4,5)) #we need 4 rows and 5 columns each
    assert(response2['b1'].shape == (4,1))
    assert(response2['W2'].shape == (3,4))
    assert(response2['b2'].shape == (3,1))
    assert(response2['W3'].shape == (1,3))
    assert(response2['b3'].shape == (1,1))
	
def test_cost_function():
    AL = np.array([[0.55,0.4,0.52,0.3]])
    y = np.array([[1,0,0,1]])
    cost = ml.cost_function(AL, y)
        
    assert(round(cost, 2) == 0.76)

def test_forward_propagation():
    params = {'W1': [[ 0.62511667, -0.5252495 , -1.19255901],
                    [ 0.07065791,  0.06861056, -0.71005542]],
             'W2': [[ 1.31945139, -1.28305008]],
             'b1': [[ 0.],
                    [ 0.]],
             'b2': [[ 0.]]}
    np.random.seed(1)
    X = np.random.randn(3,1) #note the use of randomness here
   
    cache, AL = ml.forward_propagation(X, params)
    
    assert(len(cache) == 4)
    assert(cache['Z1'].shape == (2,1))
    assert(cache['A1'].shape == (2,1))
    assert(cache['Z2'].shape == (1,1))
    assert(cache['A2'].shape == (1,1))    
    #add sth like this assert(round(float(AL[0]), 2) == 2.02)


def test_back_propagation():
    parameters = ml.initialize_parameters([1,2,1])
    epsilon = 1e-7
    lambd = 0
    X = np.array([[-0.13751202, -0.13751177, -0.13751175]])
    y = np.array([[1, 0, 0]])
    
    parameter_vector, w_shape = dict_to_vector(parameters)
    param_size = parameter_vector.shape[0] #gradient check is 
    grad_prediction = np.zeros((param_size,1))
    
    for i in range(param_size):
        theta_plus = np.copy(parameter_vector)
        theta_plus[i][0] = theta_plus[i][0] + epsilon #change theta_plus by one

        _, A_plus = ml.forward_propagation(X, vector_to_dict(theta_plus, w_shape)) #as cache is not needed here
        cost_theta_plus = ml.cost_function(A_plus, y, lambd, parameters)
        
        theta_minus = np.copy(parameter_vector)
        theta_minus[i][0] = theta_minus[i][0] - epsilon
        
        _, A_minus = ml.forward_propagation(X, vector_to_dict(theta_minus, w_shape))
        cost_theta_minus = ml.cost_function(A_minus, y, lambd, parameters)
        
        grad_prediction[i] = (cost_theta_plus - cost_theta_minus)/(2 * epsilon) #This is like taking the derivate of J. Dj/dTheta - what backprop does
    
    cache, AL = ml.forward_propagation(X, parameters) #perform forward propagation to use in backprop
    grads = ml.back_propagation(X, y, cache, parameters,lambd) #perform backpropagation
    gradVec = grads_to_vector(grads) #convert that to vector
    
    numerator = np.linalg.norm(gradVec - grad_prediction)
    denominator = np.linalg.norm(gradVec) + np.linalg.norm(grad_prediction)
    difference = numerator/denominator

    assert(difference < 2e-7)