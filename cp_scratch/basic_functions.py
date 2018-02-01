import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def convert_to_one_hot(passedArr, C):
    '''
    Converts the one hot value
    
    Arguments:
    passedArr -- Numpy array In format [[-4,-3,...4]]
    C -- The number of classifications
    
    Returns:
    array:
    The value in one hot format.
    Like:
    [[1,0,.........0],
     [0,1,.........0],
     [0,0,.........1]]
    
    '''    
    reshaped = passedArr.reshape(-1)
    newReshaped = reshaped + 4 #As there are negatives
    
    digitsLength = np.unique(newReshaped).shape[0]
    
    returnVal= np.eye(C)[newReshaped].T
    return(returnVal)

def initialize_mini_batch(X, y, batchsize=64, random=0):
    '''
    Creates minibatches from values of X and y.
    
    Parameters:
    X (numpy)
    y (numpy)

    batchsize (optional):
    Default is 64
    
    random (optional):
    Shuffles before creating minibatch if set to 1
    
    Returns:
    batches (list):
    list in the format - [[batchX1, batchY1], .... [batchXn, batchYn]]
    
    '''
    if (random == 1):
        X, y = shuffle(X,y)
        
    batches = []
    
    divide = X.shape[1] // batchsize
    
    for i in range(1,divide+1):
        batchX = X[:,:64]
        batchY = y[:,:64]
        
        batches.append((batchX,batchY))
        X = X[:,batchsize:]
        y = y[:,batchsize:] 
    
    batches.append((X,y))
    return batches

def initialize_adam(params):
        v = {}
        s = {}
        
        loop = len(params) // 2
        
        
        for i in range(1,loop+1):
            v["dW" + str(i)] = np.zeros(params['W' + str(i)].shape)
            v["db" + str(i)] = np.zeros(params['b' + str(i)].shape)
            s["dW" + str(i)] = np.zeros(params['W' + str(i)].shape)
            s["db" + str(i)] = np.zeros(params['b' + str(i)].shape)
            
        return v,s
    
def single_plot(y, title='', Xtitle='', Ytitle=''):
    fig = plt.figure();
    axes = fig.add_axes([0,0,1,1])

    axes.set_xlabel(Xtitle)
    axes.set_ylabel(Ytitle)
    axes.set_title(title)
    
    axes.plot(np.arange(1,len(y) + 1), np.log(y))