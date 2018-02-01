import cp_scratch.gradient_convertor as gc
import numpy as np

def test_dict_to_vector():
    dictionary = {'W1': [[ 0.62511667, -0.5252495 , -1.19255901],
                    [ 0.07065791,  0.06861056, -0.71005542]],
                 'W2': [[ 1.31945139, -1.28305008]],
                 'b1': [[ 0.],
                        [ 0.]],
                 'b2': [[ 0.]]}
    returnedArr,w_shape = gc.dict_to_vector(dictionary)
    
    a = np.array([[ 0.62511667],
                 [-0.5252495 ],
                 [-1.19255901],
                 [ 0.07065791],
                 [ 0.06861056],
                 [-0.71005542],
                 [ 0.        ],
                 [ 0.        ],
                 [ 1.31945139],
                 [-1.28305008],
                 [ 0.        ]])
    
    assert all(returnedArr == a)
    assert (w_shape[0][1] == 2)
    assert (w_shape[0][2] == 3)
    assert (w_shape[1][1] == 1)
    assert (w_shape[1][2] == 2)
    
    print(w_shape)
    
    
def test_vector_to_dict():
    W_shape = [(1, 2, 3), (1, 1, 2)]
    
    a = np.array([[ 0.62511667],
                 [-0.5252495 ],
                 [-1.19255901],
                 [ 0.07065791],
                 [ 0.06861056],
                 [-0.71005542],
                 [ 0.        ],
                 [ 0.        ],
                 [ 1.31945139],
                 [-1.28305008],
                 [ 0.        ]])
    
   
    
    returnedDict = gc.vector_to_dict(a, W_shape)
    
    assert (len(returnedDict) == 4)
    assert all(returnedDict['W1'][0] ==  [ 0.62511667, -0.5252495 , -1.19255901])
    assert all(returnedDict['W1'][1] ==  [ 0.07065791,  0.06861056, -0.71005542])
    assert all(returnedDict['W2'][0] ==  [ 1.31945139, -1.28305008])
    assert all(returnedDict['b2'][0] ==  [0])
    assert all(returnedDict['b1'][0] ==  [0])
    assert all(returnedDict['b1'][1] ==  [0])
    
    
def test_grads_to_vector():
    grads = {'dW1': [[ 1.31945139, -1.28305008]],
            'db1': [ 0.],
            'dW2': [[ 2.31945139, -2.28305008]],
            'db2': [ 1.]}
    
    vector = gc.grads_to_vector(grads)
    
    assert all (vector == [[ 1.31945139],
                             [-1.28305008],
                             [ 0.        ],
                             [ 2.31945139],
                             [-2.28305008],
                             [ 1.        ]])