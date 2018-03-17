from data_utils import BasicFunctions, PriceFunctions
import numpy as np
import pandas as pd

def test_convert_to_one_hot():
    bf = BasicFunctions()
    
    testArr = np.array([[-4, -3, -2, -1, -2, -1, 0, 1, 1, 2, 3, 4]])
    returned = bf.convert_to_one_hot(testArr, 9)
    
    assert(np.count_nonzero(returned[0] == 1) == 1)
    assert(returned[0][0] == 1)
    
    assert(np.count_nonzero(returned[1] == 1) == 1)
    assert(returned[1][1] == 1)
    
    assert(np.count_nonzero(returned[2] == 1) == 1)
    assert(returned[2][2] == 1)
    
    assert(np.count_nonzero(returned[3] == 1) == 1)
    assert(returned[3][3] == 1)
    
    assert(np.count_nonzero(returned[4] == 1) == 1)
    assert(returned[4][2] == 1)
    
    assert(np.count_nonzero(returned[5] == 1) == 1)
    assert(returned[5][3] == 1)
    
    assert(np.count_nonzero(returned[6] == 1) == 1)
    assert(returned[6][4] == 1)
    
    assert(np.count_nonzero(returned[7] == 1) == 1)
    assert(returned[7][5] == 1)
    
    assert(np.count_nonzero(returned[8] == 1) == 1)
    assert(returned[8][5] == 1)
    
    assert(np.count_nonzero(returned[9] == 1) == 1)
    assert(returned[9][6] == 1)
    
    assert(np.count_nonzero(returned[10] == 1) == 1)
    assert(returned[10][7] == 1)
    
    assert(np.count_nonzero(returned[11] == 1) == 1)
    assert(returned[11][8] == 1)
    
def test_initialize_mini_batch():
    bf = BasicFunctions()
    
    np.random.seed(1)
    X = np.random.randn(12288, 148)
    y = np.random.randn(1, 148)

    batches = bf.initialize_mini_batch(X,y,64)
    assert(batches[0][0].shape == (12288, 64)) #testing X
    assert(batches[1][0].shape == (12288, 64))
    assert(batches[2][0].shape == (12288, 20))
    assert(batches[0][1].shape == (1, 64)) #testing y
    assert(batches[1][1].shape == (1, 64))
    assert(batches[2][1].shape == (1, 20))
    
#test for PriceFunctions is not added now because they will be modified when price functions for other cryptos is added