import numpy as np
import pandas as pd
from datetime import datetime
from timeit import default_timer as timer

#coverts dict to vector as shown in image above and return size of w and the vector
def dict_to_vector(parameters, debug = 0):
    length = len(parameters) // 2 #as there are 2 of each. Wn and bn
    
    start_time = timer()
    
    array = []
    w_shape = []
    
    for i in range(1,length+1):
        w = [v for k, v in parameters.items() if k == ('W' + str(i))]
        w = np.asarray(w)
        w_shape.append(w.shape) #because it is needed in vector_to_dict
        w = np.reshape(w, (-1,1)) #-1 is wildcard for arrange yourself
        
        b = [v for k,v in parameters.items() if k == ('b' + str(i))]
        b = np.asarray(b)
        b = np.reshape(b, (-1,1))
        
        array.extend(w) #extend is better than append. Append keeps arrays
        array.extend(b)
    
    end_time = timer()
    diff = end_time - start_time
    
    if (debug == 1):
        print(diff * 1000)
    
    array = np.asarray(array)
    return(array, w_shape)

#convert the vector back to dictionaries of Wn and bn 
def vector_to_dict(vector, w_shape, debug=0):
    temp = {} 
    no_w = len(w_shape)

    for i in range(1,no_w+1):
        temp['W' + str(i)] = vector[:(w_shape[i-1][1] * w_shape[i-1][2])].reshape(w_shape[i-1][1], w_shape[i-1][2]) #reshape to make into 2d array by taking 1D elements
        vector = vector[(w_shape[i-1][1] * w_shape[i-1][2]):]  #removing the elements to make calculations easier      
        
        if (debug == 1):
            print(vector)
        
        temp['b' + str(i)] = vector[:w_shape[i-1][1]]
        vector = vector[w_shape[i-1][1]:]
        
        if (debug == 1):
            print(vector)
            
    return temp

#to compare the grads with the grads obtained from other ways. Arranges dW and dB in the same formate as another
def grads_to_vector(grads, debug=0):
    length = len(grads) // 2 #as there are 2 of each. Wn and bn
    
    start_time = timer()
    
    array = []
    
    for i in range(1,length+1):
        w = [v for k, v in grads.items() if k == ('dW' + str(i))]
        w = np.reshape(w, (-1,1)) #-1 is wildcard for arrange yourself
        
        b = [v for k,v in grads.items() if k == ('db' + str(i))]
        b = np.asarray(b)
        b = np.reshape(b, (-1,1))
        
        array.extend(w) #extend is better than append. Append keeps arrays
        array.extend(b)
    
    end_time = timer()
    diff = end_time - start_time
    
    if (debug == 1):
        print(diff * 1000)
    
    array = np.asarray(array)
    return(array)